# remember the approx loc of unpicked berries
# remember a low-res path

import os
from collections import deque
from typing import Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from berry_field.envs.berry_field_env import BerryFieldEnv
from berry_field.envs.utils.misc import getTrueAngles
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from torch import Tensor, nn

from make_net import (make_simple_conv1dnet, make_simple_conv2dnet,
                      make_simple_feedforward)
from print_utils import printLocals

ROOT_2_INV = 0.5**(0.5)
EPSILON = 1E-8

class Agent():
    """ states containing an approximantion of the path of the agent 
    and also computed uisng info from all seen but not-collected berries """
    def __init__(self, berryField:BerryFieldEnv, mode='train', 
                angle = 45, persistence=0.8, worth_offset=0.0, 
                noise=0.01, field_grid_size=(40,40), memory_alpha=0.9965, 
                time_memory_delta=0.005, time_memory_exp=1, nstep_transition=[1], 
                reward_patch_discovery=False, debug=False, debugDir='.temp') -> None:
        """ 
        ### parameters
        - berryField: BerryFieldEnv instance
        - mode: 'train' or 'eval'
        - angle: int (default 45)
                - the observation space divided into angular sectors
                - number of sectors is 360/angle
        - persistence: float (default 0.8)
                - the persistence of vision for the sectorized states 
        - worth_offset: float (default 0)
                - the offset in the berry worth function
        - noise: float (default 0.01)
                - uniform noise between [-noise, noise) is added to state
        - field_grid_size: tupple[int,int] (default (40,40))
                - the size of memory grid must divide field size of env
        - memory_alpha: float (default 0.9965)
         """
        printLocals('Agent', locals())
        self.istraining = mode == 'train'
        self.angle = angle
        self.persistence = persistence
        self.worth_offset = worth_offset
        self.berryField = berryField
        self.field_grid_size = field_grid_size
        self.noise = noise
        self.memory_alpha = memory_alpha
        self.time_memory_delta = time_memory_delta
        self.time_memory_exp = time_memory_exp
        self.nstep_transitions = nstep_transition # for approx n-step TD effect
        self.reward_patch_discovery = reward_patch_discovery

        # init memories and other stuff
        self.num_sectors = 360//angle        
        self.divLenX = berryField.FIELD_SIZE[0]//field_grid_size[0]
        self.divLenY = berryField.FIELD_SIZE[1]//field_grid_size[1]
        self.output_shape = self.get_output_shape()
        self._init_memories()

        self.print_information()
        
        # setup debug
        self.debugDir = debugDir
        self.debug = debug
        self.built_net = False
        if debug:
            self.debugDir = os.path.join(debugDir, 'stMakerdebug')
            if not os.path.exists(self.debugDir): os.makedirs(self.debugDir)
            self.state_debugfile = open(os.path.join(self.debugDir, 'stMakerdebugstate.txt'), 'w', 1)
            self.env_recordfile = open(os.path.join(self.debugDir, 'stMakerrecordenv.txt'), 'w', 1)

    def print_information(self):
        if not self.istraining: print('eval mode'); return
        print("""The state-transitions being appended 
            every action will be as [[state, action, sum-reward, nextState, done]] where:
            state is the one the model has taken action on,
            sum-reward is the sum of the rewards in the skip-trajectory,
            nextState is the new state after the action was repeated at most skip-steps times,
            done is wether the terminal state was reached.""")
        if self.reward_patch_discovery:
            print("Rewarding the agent for discovering new patches")
        print('agent now aware of total-juice')

    def _init_memories(self):
        memory_grid_size = self.field_grid_size[0]*self.field_grid_size[1]

        # for the approx n-step TD construct
        self.state_deque = deque(maxlen=max(self.nstep_transitions) + 1)
        self.state_deque.append([None,None,0]) # init deque

        # needed to reward patch discovery
        self.visited_patches = set() 

        # for path and berry memory
        self.path_memory = np.zeros(memory_grid_size) # aprox path
        self.berry_memory = np.zeros(memory_grid_size) # aprox place of sighting a berry
        self.time_memory = 0 # the time spent at the current block
        self.time_memory_data = np.zeros_like(self.path_memory)

        # for persistence
        self.prev_sectorized_state = np.zeros((4,self.num_sectors))
        return

    def berry_worth_function(self, sizes, distances):
        """ the reward that can be gained by pursuing a berry of given size and distance
        we note that the distances are scaled to be in range 0 to 1 by dividing by half-diag
        of observation space """
        # for computation of berry worth, can help to change 
        # the agent's preference of different sizes of berries. 
        rr, dr = self.berryField.REWARD_RATE, self.berryField.DRAIN_RATE
        worth = rr * sizes - dr * distances * self.berryField.HALFDIAGOBS
        
        # scale worth to 0 - 1 range
        min_worth, max_worth = rr * 10 - dr * self.berryField.HALFDIAGOBS, rr * 50
        worth = (worth - min_worth)/(max_worth - min_worth)

        # incorporate offset
        worth = (worth + self.worth_offset)/(1 + self.worth_offset)

        return worth

    def _reward_patch_discovery(self, skip_transitions):
        # skip_transitions -> [[observation, info, reward, done]]
        # info, reward are None at start (spawn)
        reward = 0
        for o,info,r,d in skip_transitions:
            if info is None: continue
            patch_id = info['current-patch-id']
            if patch_id is not None:
                if patch_id not in self.visited_patches:
                    self.visited_patches.add(patch_id)
                    # don't give +ve to spawn patch!!!
                    if len(self.visited_patches) > 1: reward += 1
        return reward

    def _compute_sectorized(self, raw_observation, info):
        """  """
        # a1 = np.zeros(self.num_sectors) # max-worth of each sector
        # a2 = np.zeros(self.num_sectors) # stores avg-worth of each sector
        # a3 = np.zeros(self.num_sectors) # indicates the sector with the max worthy berry
        # a4 = np.zeros(self.num_sectors) # a mesure of distance to max worthy in each sector

        # apply persistence
        a1,a2,a3,a4 = self.prev_sectorized_state * self.persistence
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

        self.prev_sectorized_state = np.array([a1,a2,a3,a4])
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
        return

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
        total_juice = info['total_juice']

        # make the state by concatenating sectorized_states and memories
        state = np.concatenate([*sectorized_states, edge_dist, patch_relative, 
                                [self.time_memory], [total_juice], 
                                self.berry_memory, self.path_memory])

        return state + np.random.uniform(-self.noise, self.noise, size=state.shape)

    def state_desc(self):
        """ returns the start and stop+1 indices for various parts of the state """
        memory_size = self.field_grid_size[0] * self.field_grid_size[1]
        state_desc = {
            'sectorized_states': [0, 4*self.num_sectors],
            'edge_dist': [4*self.num_sectors, 4*self.num_sectors+4],
            'patch_relative': [4*self.num_sectors+4, 4*self.num_sectors+4+1],
            'time_memory': [4*self.num_sectors+4+1, 4*self.num_sectors+4+2],
            'total_juice': [4*self.num_sectors+4+2, 4*self.num_sectors+4+3],
            'berry_memory': [4*self.num_sectors+4+3, 4*self.num_sectors+4+3 + memory_size],
            'path_memory': [4*self.num_sectors+4+3 + memory_size, 4*self.num_sectors+4+3 + 2*memory_size]
        }
        return state_desc

    def get_output_shape(self):
        try: return self.output_shape
        except: self._computeState(None, None, None, None).shape

    def makeState(self, skip_trajectory, action_taken):
        """ skip trajectory is a sequence of [[next-observation, info, reward, done],...] """
        final_state = self._computeState(*skip_trajectory[-1])

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
        # compute the sum of reward in the skip-trajectory
        reward = sum([r for o,i,r,d in skip_trajectory])
        reward += self._reward_patch_discovery(skip_trajectory) if self.reward_patch_discovery else 0
        done = skip_trajectory[-1][-1]

        # update dequeue
        self.state_deque.append([state, action, reward + self.state_deque[-1][-1]])    
        transitions = []

        # transitions for crude nstep TD approximations
        for n in self.nstep_transitions:
            if len(self.state_deque) >= n+1:
                reward0 = self.state_deque[-n-1][-1]
                oldstate, oldaction, reward1 = self.state_deque[-n]
                sum_reward = max(min(reward1-reward0,2),-2) # clipping so that this doesnot wreck priority buffer
                transition = [oldstate, oldaction, sum_reward, nextState, done]
                transitions.append(transition)
        return transitions

    def getNet(self, TORCH_DEVICE, debug=False,
            feedforward = dict(linearsDim = [16,8], lreluslope=0.1),
            memory_conv = dict(channels = [8,16,16], kernels = [4,3,3], 
                strides = [2,2,2], paddings = [3,2,1], maxpkernels = [2,0,2],
                padding_mode='zeros', lreluslope=0.1, add_batchnorms=False),
            final_stage = dict(infeatures=136, linearsDim = [16,16], 
                lreluslope=0.1)):
        """ create and return the model (a duelling net)"""
        num_sectors = self.num_sectors
        memory_shape = self.field_grid_size
        memory_size = memory_shape[0]*memory_shape[1]
        outDims = self.berryField.action_space.n

        class net(nn.Module):
            def __init__(self):
                super(net, self).__init__()

                # build the feed-forward network -> sectors, edge, patch-relative & time-memory
                self.feedforward = make_simple_feedforward(infeatures=4*num_sectors+7, **feedforward)

                # build the conv-networks
                self.memory_conv1 = make_simple_conv2dnet(inchannel=1, **memory_conv)
                self.memory_conv2 = make_simple_conv2dnet(inchannel=1, **memory_conv)

                # build the final stage
                self.final_stage = make_simple_feedforward(**final_stage)
                
                # for action advantage estimates
                self.valueL = nn.Linear(final_stage['linearsDim'][-1], 1)
                self.actadvs = nn.Linear(final_stage['linearsDim'][-1], outDims)

                # indices to split at
                self.ffpart = (0, 4*num_sectors+7)
                self.mempart = (self.ffpart[1], self.ffpart[1]+2*memory_shape[0]*memory_shape[1])

                print('seperate conv-nets for berry and path memory')

            def forward(self, input:Tensor):

                # split and reshape input
                if debug: print(input.shape)
                feedforward_part = input[:,self.ffpart[0]:self.ffpart[1]]
                memory_part = input[:,self.mempart[0]:self.mempart[1]]

                # conv2d requires 4d inputs
                if debug: print('memory_part:', memory_part.shape)
                bery_memory = memory_part[:,:memory_size].reshape((-1,1,*memory_shape))
                path_memory = memory_part[:,memory_size:].reshape((-1,1,*memory_shape))

                # process feedforward_part
                if debug: print('\nfeedforward_part',feedforward_part.shape)
                for layer in self.feedforward:
                    feedforward_part = layer(feedforward_part)        

                # process path_memory
                if debug: print('\npath_memory', path_memory.shape)
                for i, layer in enumerate(self.memory_conv1):
                    path_memory = layer(path_memory)
                    if debug: print(layer.__class__.__name__,i,path_memory.shape)

                # process bery_memory
                if debug: print('\nberry_memory', bery_memory.shape)
                for i, layer in enumerate(self.memory_conv2):
                    bery_memory = layer(bery_memory)
                    if debug: print(layer.__class__.__name__,i,bery_memory.shape)

                # merge all
                path_memory = torch.flatten(path_memory, start_dim=1)
                bery_memory = torch.flatten(bery_memory, start_dim=1)
                if debug: print('\nfor concat',feedforward_part.shape, 
                                bery_memory.shape, path_memory.shape)

                # process merged features
                concat = torch.cat([feedforward_part, bery_memory, path_memory], dim=1)
                if debug: print('concat',concat.shape)
                for layer in self.final_stage:
                    concat = layer(concat)

                value = self.valueL(concat)
                advs = self.actadvs(concat)
                qvalues = value + (advs - advs.mean())
                
                return qvalues
        
        self.nnet = net().to(TORCH_DEVICE)
        self.built_net = True
        print('total-params: ', sum(p.numel() for p in self.nnet.parameters() if p.requires_grad))
        return self.nnet

    def showDebug(self, nnet:Union[nn.Module,None]=None, debugDir = None, f=20, gif=False):
        
        # close the log files if not already closed
        if self.debug and not self.state_debugfile.closed:
            self.state_debugfile.write('end')
            self.state_debugfile.close()
        if self.debug and not self.env_recordfile.closed:
            self.env_recordfile.write('end')
            self.env_recordfile.close()
        
        # init the debug directory
        if not debugDir: debugDir = self.debugDir
        else: debugDir = os.path.join(debugDir, 'stMakerdebug')

        # init the gif file
        if gif:
            giffile = imageio.get_writer(f'{debugDir}/debug.gif')

        # move nnet to cpu
        if nnet: nnet.cpu()

        fig, ax = plt.subplots(2,3, figsize=(15, 10))
        plt.tight_layout(pad=5)
        staterecord = open(os.path.join(debugDir, 'stMakerdebugstate.txt'), 'r')
        envrecord = open(os.path.join(debugDir, 'stMakerrecordenv.txt'), 'r')
        action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'X']
        
        itter = -1
        while True:
            line = staterecord.readline()
            line2 = envrecord.readline()
            if line == 'end': break
            itter += 1
            if itter % f != 0: continue

            state = np.array(eval('[' + line[:-1].replace(' ', ',') + ']'), dtype=float)
            agent_and_berries = np.array(eval('[' + line2[:-1].replace(' ', ',') + ']'), dtype=float).reshape(-1,3)

            sectorized_states = state[:4*self.num_sectors].reshape(4,self.num_sectors)
            edge_dist = state[4*self.num_sectors: 4*self.num_sectors+4]
            patch_relative = state[4*self.num_sectors+4:4*self.num_sectors+4+1]
            time_memory = state[4*self.num_sectors+4+1]
            total_juice = state[4*self.num_sectors+4+2]
            memories = state[4*self.num_sectors+4+3:]
            berry_memory = memories[:self.field_grid_size[0]*self.field_grid_size[1]].reshape(self.field_grid_size)
            path_memory = memories[self.field_grid_size[0]*self.field_grid_size[1]:].reshape(self.field_grid_size)

            berries = agent_and_berries[1:]
            agent = agent_and_berries[0]
            w,h = self.berryField.OBSERVATION_SPACE_SIZE
            W, H = self.berryField.FIELD_SIZE

            ax[0][0].imshow(sectorized_states)
            ax[0][1].bar([0,1,2],[total_juice, patch_relative, time_memory], [1,1,1])
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
            ax[0][1].set_xticklabels(["","","total-juice","","patch-rel","","time-mem"]) 
            ax[0][2].set_title('measure of dist-from-edge')
            ax[0][2].set_xticklabels(["","left","right","top","bottom"]) 
            ax[1][0].set_title('berry-memory (avg-worth)')
            ax[1][1].set_title('path-memory')
            ax[0][1].set_ylim(top=1)
            if not nnet: ax[1][2].set_title(f'env-record')

            if gif: 
                fig.savefig(f'{debugDir}/tmpimg.png')
                img = imageio.imread(f'{debugDir}/tmpimg.png')
                giffile.append_data(img)
                os.remove(f'{debugDir}/tmpimg.png')

            plt.pause(0.001)
            
            for b in ax: 
                for a in b: a.clear() 
            
        plt.show(); plt.close()
        staterecord.close(); envrecord.close()
        if gif: giffile.close()
