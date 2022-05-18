# remember the approx loc of unpicked berries
# remember a low-res path

import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from berry_field.envs.berry_field_env import BerryFieldEnv
from berry_field.envs.utils.misc import getTrueAngles
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from torch import Tensor, nn
from print_utils import printLocals

from make_net import make_simple_conv1dnet, make_simple_conv2dnet, make_simple_feedforward

ROOT_2_INV = 0.5**(0.5)
EPSILON = 1E-8

class Agent():
    """ states containing an approximantion of the path of the agent 
    and also computed uisng info from all seen but not-collected berries """
    def __init__(self, berryField:BerryFieldEnv, mode='train', field_grid_size=(40,40), 
                    angle = 45, persistence=0.7, worth_offset=0.0, noise=0.02, positive_emphasis=True,
                    emphasis_mode= 'replace', memory_alpha=0.995, time_memory_delta=0.005, 
                    time_memory_exp=1, disjoint=False, debug=False, debugDir='.temp') -> None:
        """ mode is required to assert whether it is required to make transitions """
        printLocals('Agent', locals())
        self.istraining = mode == 'train'
        self.angle = angle
        self.persistence = persistence
        self.worth_offset = worth_offset
        self.berryField = berryField
        self.field_grid_size = field_grid_size
        self.positive_emphasis = positive_emphasis
        self.append_mode = emphasis_mode == 'append'
        self.noise = noise
        self.memory_alpha = memory_alpha
        self.time_memory_delta = time_memory_delta
        self.time_memory_exp = time_memory_exp
        self.disjoint = disjoint

        # init memories and other stuff
        self.num_sectors = 360//angle        
        self.state_transitions = []
        self.output_shape = None
        self._init_memories()

        self.divLenX = berryField.FIELD_SIZE[0]//field_grid_size[0]
        self.divLenY = berryField.FIELD_SIZE[1]//field_grid_size[1]

        if self.positive_emphasis:
            print(f'''positive rewards are now emphasised in the state-transitions
            Once a berry is encountered (say at index i), new transitions of the following
            description will also be appended (if emphasis_mode = 'append') or the entries 
            will be replaced: all the transitions k < i such that the sum of reward from
            k to i is positive will have the next-state replaced by the state at transition
            at index i. And the rewards will also be replaced by the summation from k to i.
            currently, emphasis-mode is {'append' if self.append_mode else 'replace'}.
            if disjoint=True, then k is limited to the index of the last berry seen
            currently disjoint behaviour is set to {self.disjoint}\n''')
        
        # setup debug
        self.debugDir = debugDir
        self.debug = debug
        self.built_net = False
        if debug:
            self.debugDir = os.path.join(debugDir, 'stMakerdebug')
            if not os.path.exists(self.debugDir): os.makedirs(self.debugDir)
            self.state_debugfile = open(os.path.join(self.debugDir, 'stMakerdebugstate.txt'), 'w', 1)
            self.env_recordfile = open(os.path.join(self.debugDir, 'stMakerrecordenv.txt'), 'w', 1)
            # self.qvals_debugfile = open(os.path.join(self.debugDir, 'stMakerdebugqvals.txt'), 'w', 1)


    def _init_memories(self):
        memory_grid_size = self.field_grid_size[0]*self.field_grid_size[1]
        self.path_memory = np.zeros(memory_grid_size) # aprox path
        self.berry_memory = np.zeros(memory_grid_size) # aprox place of sighting a berry
        self.time_memory = 0 # the time spent at the current block
        self.time_memory_data = np.zeros_like(self.path_memory)
        self.prev_sectorized = np.zeros((4,self.num_sectors))

    # for computation of berry worth, can help to change 
    # the agent's preference of different sizes of berries. 
    def berry_worth_function(self, sizes, distances):
        """ the reward that can be gained by pursuing a berry of given size and distance
        we note that the distances are scaled to be in range 0 to 1 by dividing by half-diag
        of observation space """
        rr, dr = self.berryField.REWARD_RATE, self.berryField.DRAIN_RATE
        worth = rr * sizes - dr * distances * self.berryField.HALFDIAGOBS
        
        # scale worth to 0 - 1 range
        min_worth, max_worth = rr * 10 - dr * self.berryField.HALFDIAGOBS, rr * 50
        worth = (worth - min_worth)/(max_worth - min_worth)

        # incorporate offset
        worth = (worth + self.worth_offset)/(1 + self.worth_offset)

        return worth

    def _compute_sectorized(self, raw_observation, info):
        """  """
        # a1 = np.zeros(self.num_sectors) # max-worth of each sector
        # a2 = np.zeros(self.num_sectors) # stores avg-worth of each sector
        # a3 = np.zeros(self.num_sectors) # indicates the sector with the max worthy berry
        # a4 = np.zeros(self.num_sectors) # a mesure of distance to max worthy in each sector

        # apply persistence
        a1,a2,a3,a4 = self.prev_sectorized * self.persistence
        total_worth = 0

        if len(raw_observation) > 0:
            sizes = raw_observation[:,2]
            dist = np.linalg.norm(raw_observation[:,:2], axis=1) + EPSILON
            directions = raw_observation[:,:2]/dist[:,None]
            angles = getTrueAngles(directions)
            
            dist = ROOT_2_INV*dist # range in 0 to 1
            maxworth = float('-inf')
            maxworth_idx = -1
            for x in range(0,360,self.angle):
                sectorL, sectorR = (x-self.angle/2)%360, (x+self.angle/2)
                if sectorL < sectorR:
                    args = np.argwhere((angles>=sectorL)&(angles<=sectorR))
                else:
                    args = np.argwhere((angles>=sectorL)|(angles<=sectorR))
                
                if args.shape[0] > 0: 
                    idx = x//self.angle
                    _sizes = sizes[args]
                    _dists = dist[args]
                    # max worthy
                    worthinesses= self.berry_worth_function(_sizes,_dists)
                    maxworthyness_idx = np.argmax(worthinesses)
                    a1[idx] = worthyness = worthinesses[maxworthyness_idx]
                    a2[idx] = np.average(worthinesses)
                    a4[idx] = 1 - _dists[maxworthyness_idx]
                    total_worth += sum(worthinesses)
                    if worthyness > maxworth:
                        maxworth_idx = idx
                        maxworth = worthyness    
            if maxworth_idx > -1: a3[maxworth_idx]=1 
        
        avg_worth = total_worth/len(raw_observation) if len(raw_observation) > 0 else 0

        self.prev_sectorized = np.array([a1,a2,a3,a4])
        return [a1,a2,a3,a4], avg_worth

    def _update_memories(self, info, avg_worth):
        """ update the path-memory and berry memory """
        x,y = info['position']

        # if agent touches the top/left edge of field, this will cause
        # an Index error. Subtract a small amount to avoid this
        if x == self.berryField.FIELD_SIZE[0]: x -= EPSILON
        if y == self.berryField.FIELD_SIZE[1]: y -= EPSILON

        # find the corresponding index in memory
        x //= self.divLenX
        y //= self.divLenY
        index = x + y*self.field_grid_size[0]
        index = int(index)

        # decay the path memory and updaate
        self.path_memory *= self.memory_alpha 
        self.path_memory[index] = 1
        
        # update berry memory
        self.berry_memory[index] = 0.2*avg_worth + 0.8*self.berry_memory[index]

        # decay time memory and update time_memory
        self.time_memory_data *= 1-self.time_memory_delta
        self.time_memory_data[index] += self.time_memory_delta
        self.time_memory = min(1,self.time_memory_data[index])**self.time_memory_exp

    def _computeState(self, raw_observation, info, reward, done) -> np.ndarray:
        """ makes a state from the observation and info. reward, done are ignored """
        # if this is the first state (a call to BerryFieldEnv.reset) -> marks new episode
        if info is None: # reinit memory and get the info and raw observation from berryField
            self._init_memories()
            raw_observation = self.berryField.raw_observation()
            info = self.berryField.get_info()

        # the total-worth is also representative of the percived goodness of observation
        sectorized_states, avg_worth = self._compute_sectorized(raw_observation, info)

        # update memories
        self._update_memories(info, avg_worth)

        # other extra information
        edge_dist = info['scaled_dist_from_edge']
        patch_relative = info['patch-relative']

        # make the state by concatenating sectorized_states and memories
        state = np.concatenate([*sectorized_states, edge_dist, patch_relative, 
                                [self.time_memory], self.berry_memory, self.path_memory])

        return state + np.random.uniform(-self.noise, self.noise, size=state.shape)

    def _compute_transitons_n_finalstate(self, skip_trajectory, action_taken):
        self.state_transitions = []
        current_state = self._computeState(*skip_trajectory[0])
        for i in range(1, len(skip_trajectory)):
            reward, done = skip_trajectory[i][2:]
            next_state = self._computeState(*skip_trajectory[i])
            self.state_transitions.append([current_state, action_taken, reward, next_state, done])
            current_state = next_state
        
        if self.positive_emphasis:# more emphasis on positive rewards
            # find where berries were encountered
            berry_indices = [0] + [i for i,x in enumerate(self.state_transitions) if x[2] > 0]
            # for each k: idx1 < k < idx2 for consequitive idx1,idx2 in berry indices
            # append transitions with start-state at k and next-state at idx2 (goal)
            # if the summed reward from k to idx2 is positive
            for i in range(1, len(berry_indices)):
                idx1, idx2 = berry_indices[i-1:i+1]
                reward_b = self.state_transitions[idx2][2]
                good_state = self.state_transitions[idx2][3]
                for k in range(idx2-1, idx1 if self.disjoint else -1, -1):
                    s, a, r, ns, d = self.state_transitions[k]
                    reward_b += r
                    if reward_b < 0: break
                    if self.append_mode: self.state_transitions.append([s,a,reward_b,good_state,d])
                    else: self.state_transitions[k][2:4] = [reward_b,good_state]
        
        return current_state # the final state

    def state_desc(self):
        """ returns the start and stop+1 indices for various parts of the state """
        memory_size = self.field_grid_size[0] * self.field_grid_size[1]
        state_desc = {
            'sectorized_states': [0, 4*self.num_sectors],
            'edge_dist': [4*self.num_sectors, 4*self.num_sectors+4],
            'patch_relative': [4*self.num_sectors+4, 4*self.num_sectors+4+1],
            'time_memory': [4*self.num_sectors+4+1, 4*self.num_sectors+4+2],
            'berry_memory': [4*self.num_sectors+4+1, 4*self.num_sectors+4+1 + memory_size],
            'path_memory': [4*self.num_sectors+4+1 + memory_size, 4*self.num_sectors+4+1 + memory_size]
        }
        return state_desc

    def get_output_shape(self):
        if not self.output_shape:
            self.output_shape = self._computeState(None, None, None, None).shape
        return self.output_shape

    def makeState(self, skip_trajectory, action_taken):
        """ skip trajectory is a sequence of [[next-observation, info, reward, done],...] """
        if not self.istraining: 
            final_state = self._computeState(*skip_trajectory[-1])
        else:
            final_state = self._compute_transitons_n_finalstate(skip_trajectory, action_taken)

        # debug
        if self.debug: 
            agent, berries = self.berryField.get_human_observation()
            np.savetxt(self.state_debugfile, [final_state])
            np.savetxt(self.env_recordfile, [np.concatenate([agent, *berries[:,:3]])])

        return final_state
           
    def makeStateTransitions(self, skip_trajectory, state, action, nextState):
        """ get the state-transitions - these are already computed in the
        makeState function when mode = 'train'. All inputs to  makeStateTransitions
        are ignored."""
        return self.state_transitions
    
    def getNet(self, TORCH_DEVICE, debug=False,
                linearsDim = [4], # for edge, patch-relative & time-memory
                sector_conv = dict(channels = [2,1], kernels = [3,3], 
                    strides = [1,1], paddings = [1,1], maxpkernels = [],
                    padding_mode='circular'),
                memory_conv = dict(channels = [8,8,16], kernels = [4,3,2], 
                    strides = [2,2,2], paddings = [3,3,1], maxpkernels = [2,2],
                    padding_mode='zeros'),
                final_linears = [204, 16]):
        """ create and return the model (a duelling net)"""
        num_sectors = self.num_sectors
        memory_shape = self.field_grid_size
        outDims = self.berryField.action_space.n

        class net(nn.Module):
            def __init__(self):
                super(net, self).__init__()

                # build the feed-forward network -> edge, patch-relative & time-memory
                self.feedforward = make_simple_feedforward(infeatures=6, linearsDim=linearsDim)

                # build the conv-networks
                self.sector_conv = make_simple_conv1dnet(inchannel=4, **sector_conv)
                self.memory_conv = make_simple_conv2dnet(inchannel=1, **memory_conv)

                # build the final stage
                self.final_stage = make_simple_feedforward(final_linears[0], final_linears[1:])
                
                # for action advantage estimates
                self.valueL = nn.Linear(final_linears[-1], 1)
                self.actadvs = nn.Linear(final_linears[-1], outDims)

                # indices to split at
                self.sectorpart = (0, 4*num_sectors)
                self.ffpart = (self.sectorpart[1], self.sectorpart[1]+6)
                self.mempart = (self.ffpart[1], self.ffpart[1]+2*memory_shape[0]*memory_shape[1])

            def forward(self, input:Tensor):

                # split and reshape input
                if debug: print(input.shape)
                sector_part = input[:,self.sectorpart[0]:self.sectorpart[1]]
                feedforward_part = input[:,self.ffpart[0]:self.ffpart[1]]
                memory_part = input[:,self.mempart[0]:self.mempart[1]]

                # conv2d requires 4d inputs
                sector_part = sector_part.reshape((-1,4,num_sectors))
                memory_part = memory_part.reshape((-1,1,2*memory_shape[0], memory_shape[1]))

                # get feed-forward output
                if debug: print('\nfeedforward_part',feedforward_part.shape)
                for layer in self.feedforward:
                    feedforward_part = layer(feedforward_part)

                # process sectors
                if debug: print('\nsector_conv',sector_part.shape)
                for i,layer in enumerate(self.sector_conv):
                    sector_part = layer(sector_part)
                    if debug: print(layer.__class__.__name__,i,sector_part.shape)           

                # process memory_part
                if debug: print('\nmemory_part', memory_part.shape)
                for i, layer in enumerate(self.memory_conv):
                    memory_part = layer(memory_part)
                    if debug: print(layer.__class__.__name__,i,memory_part.shape)

                # merge all and process
                memory_part = torch.flatten(memory_part, start_dim=1)
                sector_part = torch.flatten(sector_part, start_dim=1)
                if debug: print('\nfor concat',feedforward_part.shape,sector_part.shape,memory_part.shape)
                concat = torch.cat([memory_part, sector_part, feedforward_part], dim=1)

                if debug: print('concat',concat.shape)
                for layer in self.final_stage:
                    concat = layer(concat)

                value = self.valueL(concat)
                advs = self.actadvs(concat)
                qvalues = value + (advs - advs.mean())
                
                return qvalues
        
        nnet = net()
        nnet.to(TORCH_DEVICE)
        self.built_net = True
        print('total-params: ', sum(p.numel() for p in nnet.parameters() if p.requires_grad))
        return nnet


    def showDebug(self, nnet:Union[nn.Module,None]=None, debugDir = None):
        
        # close the log files if not already closed
        if self.debug and not self.state_debugfile.closed:
            self.state_debugfile.write('end')
            self.state_debugfile.close()
        if self.debug and not self.env_recordfile.closed:
            self.env_recordfile.write('end')
            self.env_recordfile.close()
        
        if not debugDir: debugDir = self.debugDir

        # move nnet to cpu
        if nnet: nnet.cpu()

        fig, ax = plt.subplots(2,3, figsize=(15, 10))
        plt.tight_layout(pad=5)
        staterecord = open(os.path.join(debugDir, 'stMakerdebugstate.txt'), 'r')
        envrecord = open(os.path.join(debugDir, 'stMakerrecordenv.txt'), 'r')
        action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'X']
        
        while True:
            line = staterecord.readline()
            line2 = envrecord.readline()
            if line == 'end': break

            state = np.array(eval('[' + line[:-1].replace(' ', ',') + ']'), dtype=float)
            agent_and_berries = np.array(eval('[' + line2[:-1].replace(' ', ',') + ']'), dtype=float).reshape(-1,3)

            sectorized_states = state[:4*self.num_sectors].reshape(4,self.num_sectors)
            edge_dist = state[4*self.num_sectors: 4*self.num_sectors+4]
            patch_relative = state[4*self.num_sectors+4:4*self.num_sectors+4+1]
            time_memory = state[4*self.num_sectors+4+1]
            memories = state[4*self.num_sectors+4+2:]
            berry_memory = memories[:self.field_grid_size[0]*self.field_grid_size[1]].reshape(self.field_grid_size)
            path_memory = memories[self.field_grid_size[0]*self.field_grid_size[1]:].reshape(self.field_grid_size)

            berries = agent_and_berries[1:]
            agent = agent_and_berries[0]
            w,h = self.berryField.OBSERVATION_SPACE_SIZE
            W, H = self.berryField.FIELD_SIZE

            ax[0][0].imshow(sectorized_states)
            ax[0][1].bar([0,1,2],[1,patch_relative, time_memory], [0,1,1])
            ax[0][2].bar([*range(4)],edge_dist)
            ax[1][0].imshow(berry_memory)
            ax[1][1].imshow(path_memory)

            # draw the berry-field
            ax[1][2].add_patch(Rectangle((agent[0]-w/2, agent[1]-h/2), w,h, fill=False))
            ax[1][2].add_patch(Rectangle((agent[0]-w/2-30,agent[1]-h/2-30), w+60,h+60, fill=False))
            ax[1][2].scatter(x=berries[:,0], y=berries[:,1], s=berries[:,2], c='r')
            ax[1][2].scatter(x=agent[0], y=agent[1], s=agent[2], c='black')
            if agent[0]-w/2 < 0: ax[1][2].add_patch(Rectangle((0, agent[1] - h/2), 1, h, color='blue'))
            if agent[1]-h/2 < 0: ax[1][2].add_patch(Rectangle((agent[0] - w/2, 0), w, 1, color='blue'))
            if W-agent[0]-w/2<0: ax[1][2].add_patch(Rectangle((W, agent[1] - h/2), 1, h, color='blue'))
            if H-agent[1]-h/2<0: ax[1][2].add_patch(Rectangle((agent[0] - w/2, H), w, 1, color='blue'))

            # compute q-values and plot qvals
            if nnet: 
                originalqvals = nnet(torch.tensor([state], dtype=torch.float32)).detach()[0].numpy()
                maxidx = np.argmax(originalqvals)
                ax[1][2].text(agent[0]+20, agent[1]+20, f'q:{originalqvals[maxidx]:.2f}:{action_names[maxidx]}')

                # add action-advs circles
                colors = (originalqvals-min(originalqvals))/(max(originalqvals)-min(originalqvals)+EPSILON)
                for angle in range(0, 360, self.angle):
                    rad = 2*np.pi * (angle/360)
                    x,y = 100*np.sin(rad), 100*np.cos(rad)
                    c = colors[angle//self.angle]
                    ax[1][2].add_patch(Circle((agent[0]+x, agent[1]+y), 20, color=(c,c,0,1)))

                # set title
                str_qvals = [f"{np.round(x,2):.2f}" for x in originalqvals.tolist()]
                meanings = [action_names[i]+' '*(len(qv)-len(action_names[i])) for i,qv in enumerate(str_qvals)]
                ax[1][2].set_title(f'env-record with q-vals plot\nqvals: {" ".join(str_qvals)}\n       {" ".join(meanings)}')

            # titles and ticks
            ax[0][0].set_title('sectorized states')
            ax[0][1].set_title('measure of patch-center-dist')
            ax[0][1].set_xticklabels(["","","","patch-rel","","time-mem"]) 
            ax[0][2].set_title('measure of dist-from-edge')
            ax[0][2].set_xticklabels(["","left","right","top","bottom"]) 
            ax[1][0].set_title('berry-memory (avg-worth)')
            ax[1][1].set_title('path-memory')
            if not nnet: ax[1][2].set_title(f'env-record')

            plt.pause(0.001)
            
            for b in ax: 
                for a in b: a.clear() 
            
        plt.show(); plt.close()
        staterecord.close(); envrecord.close()

