import os
from collections import deque
from typing import Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchviz
from berry_field.envs.berry_field_env import BerryFieldEnv
from berry_field.envs.utils.misc import getTrueAngles
from gym.spaces import Discrete
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from torch import Tensor, nn

from utils import printLocals
from utils import random_exploration
from utils.nn_utils import (make_simple_conv1dnet, make_simple_conv2dnet,
                            make_simple_feedforward)

ROOT_2_INV = 0.5**(0.5)
EPSILON = 1E-8

class Agent():
    """ states containing an approximantion of the path of the agent 
    and also computed uisng info from all seen but not-collected berries """
    def __init__(self, berryField:BerryFieldEnv, mode='train', 
                angle = 45, persistence=0.8, worth_offset=0.0, 
                noise=0.01, nstep_transition=[1], positive_emphasis=0,
                reward_patch_discovery=True, 
                add_exploration = True,
                time_memory_delta=0.01, time_memory_exp=1.0,
                render=False, renderstep=10, debug=False, debugDir='.temp') -> None:
        """ 
        ### parameters
        - berryField: BerryFieldEnv instance
        - mode: 'train' or 'eval'
        - angle: int (default 45)
                - the observation space divided into angular sectors
                - number of sectors is 360/angle
        - persistence: float (default 0.8)
                - the persistence of vision for the sectorized states
                and the time-memory 
        - worth_offset: float (default 0)
                - the offset in the berry worth function
        - noise: float (default 0.01)
                - uniform noise between [-noise, noise) is added to state
        - nstep_transition: list[int] (default [1])
                - for each int 'k' in the list, a transition is appended
                such that the state and next state are seperated by
                'k' actions. The reward is summed.
        - positive_emphasis: int (default 0)
                - state transitions with positive reward are 
                repeatedly output for positive_emphasis number of
                times in makeStateTransitions
        - reward_patch_discovery: bool (default True)
                - add +1 reward on discovering a new patch
        - add_exploration: bool (default True)
                - adds exploration-subroutine as an action
        - time_memory_delta: float (default 0.01)
                - increment the time of the current block
                by time_memory_delta for each step in the block
        - time_memory_exp: float (default 1.0)
                - raise the stored time memory for the current block
                to time_memory_exp and feed to agent's state
        - render: bool (default False)
                - wether to render the agent 
        - renderstep: int (defautl False)
                - render the agent every renderstep step
         """
        printLocals('Agent', locals())
        self.istraining = mode == 'train'
        self.angle = angle
        self.persistence = persistence
        self.worth_offset = worth_offset
        self.berryField = berryField
        self.noise = noise
        self.nstep_transitions = nstep_transition # for approx n-step TD effect
        self.reward_patch_discovery = reward_patch_discovery
        self.positive_emphasis = positive_emphasis
        self.add_exploration= add_exploration
        self.time_memory_delta = time_memory_delta
        self.time_memory_exp = time_memory_exp
        self.render = render
        self.renderstep = renderstep

        # init memories and other stuff
        self.num_sectors = 360//angle        
        self.output_shape = self.get_output_shape()
        self._init_memories()
        self.berryField.step = self.env_step_wrapper(self.berryField, render, renderstep)

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
        if self.add_exploration:
            print("Exploration subroutine added")
        print('agent now aware of total-juice')

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

    def env_step_wrapper(self, berryField:BerryFieldEnv, render, renderstep):
        """ kinda magnifies rewards by 2/(berry_env.REWARD_RATE*MAXSIZE)
        for better gradients..., also rewards are clipped between 0 and 2 """
        print('with living cost, rewards scaled by 2/(berryField.REWARD_RATE*MAXSIZE)')
        print('rewards are clipped between 0 and 2')
        
        MAXSIZE = max(berryField.berry_collision_tree.boxes[:,2])
        scale = 2/(berryField.REWARD_RATE*MAXSIZE)
        cur_n = berryField.action_space.n
        berry_env_step = berryField.step

        # add the exploration subroutine to env action space
        if self.add_exploration:
            print('Exploration subroutine as an action')
            exploration_step = random_exploration(berryField, render, renderstep)
            berryField.action_space = Discrete(cur_n+1)

        # some stats to track
        actual_steps = 0
        episode = 0
        action_counts = {i:0 for i in range(berryField.action_space.n)}
        def step(action):
            nonlocal actual_steps, episode

            # execute the action
            steps, state, reward, done, info = exploration_step() \
                if self.add_exploration and action == cur_n else \
                    (1, *berry_env_step(action))

            # update stats
            actual_steps += steps
            action_counts[action] += 1

            # modify reward
            reward = scale*reward
            reward = min(max(reward, 0), 2)
            if self.reward_patch_discovery: 
                reward += self._reward_patch_discovery(info)
            
            # print stuff and reset stats
            if done: 
                print(f'\n=== episode:{episode} Env-steps-taken:{actual_steps}')
                print('action_counts:',action_counts)
                print('picked: ', berryField.get_numBerriesPicked())
                actual_steps = 0
                episode+=1
                for k in action_counts:action_counts[k]=0
            
            # render if asked
            if render and actual_steps%renderstep==0:
                berryField.render()
            
            return state, reward, done, info
        return step

    def _init_memories(self):
        # for the approx n-step TD construct
        self.state_deque = deque(maxlen=max(self.nstep_transitions) + 1)
        self.state_deque.append([None,None,0]) # init deque

        # needed to reward patch discovery
        self.visited_patches = set() 

        # for persistence
        self.prev_sectorized_state = np.zeros((4,self.num_sectors))

        # for time memory
        self.time_memory = 0 # the time spent at the current block
        self.time_memory_data = np.zeros((200,200))
        return

    def _reward_patch_discovery(self, info):
        # info, reward are None at start (spawn)
        reward = 0
        if info is None: return reward
        patch_id = info['current-patch-id']
        if (patch_id is not None) and (patch_id not in self.visited_patches):
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
        x,y = info['position']
        if x == self.berryField.FIELD_SIZE[0]: x -= EPSILON
        if y == self.berryField.FIELD_SIZE[1]: y -= EPSILON

        # decay time memory and update time_memory
        mem_x, mem_y = self.time_memory_data.shape
        x = int(x//(self.berryField.FIELD_SIZE[0]//mem_x))
        y = int(y//(self.berryField.FIELD_SIZE[1]//mem_y))
        self.time_memory_data *= 1-self.time_memory_delta
        self.time_memory_data[x][y] += self.time_memory_delta
        current_time = min(1,self.time_memory_data[x][y])**self.time_memory_exp
        self.time_memory = self.time_memory*self.persistence +\
                            (1-self.persistence)*current_time
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
        # rel_p = info['relative_coordinates']

        # make the state by concatenating sectorized_states and memories
        state = np.concatenate([*sectorized_states, edge_dist, patch_relative, 
                                [total_juice], [self.time_memory]])

        return state + np.random.uniform(-self.noise, self.noise, size=state.shape)

    def state_desc(self):
        """ returns the start and stop+1 indices for various parts of the state """
        state_desc = {
            'sectorized_states': [0, 4*self.num_sectors],
            'edge_dist': [4*self.num_sectors, 4*self.num_sectors+4],
            'patch_relative': [4*self.num_sectors+4, 4*self.num_sectors+4+1],
            'total_juice': [4*self.num_sectors+4+1, 4*self.num_sectors+4+2],
            'time_memory': [4*self.num_sectors+4+2, 4*self.num_sectors+4+3],
            # 'relative_coordinates':[4*self.num_sectors+7, 4*self.num_sectors+8]
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
        done = skip_trajectory[-1][-1]

        # update dequeue
        self.state_deque.append([state, action, reward + self.state_deque[-1][-1]])    
        transitions = []

        # transitions for crude nstep TD approximations
        for n in self.nstep_transitions:
            if len(self.state_deque) >= n+1:
                reward0 = self.state_deque[-n-1][-1]
                oldstate, oldaction, reward1 = self.state_deque[-n]
                sum_reward = reward1-reward0
                # sum_reward = max(min(reward1-reward0,2),-2) # clipping so that this doesnot wreck priority buffer
                transition = [oldstate, oldaction, sum_reward, nextState, done]
                if self.positive_emphasis and sum_reward > 0:
                    transitions.extend([transition]*self.positive_emphasis)
                else: transitions.append(transition)
        return transitions

    def getNet(self, TORCH_DEVICE, debug=False,
            feedforward = dict(infeatures=39, linearsDim = [32,16], lreluslope=0.1),
            final_stage = dict(infeatures=16, linearsDim = [8], 
                lreluslope=0.1), saveVizpath=None):
        """ create and return the model (a duelling net)"""
        num_sectors = self.num_sectors
        outDims = self.berryField.action_space.n

        class net(nn.Module):
            def __init__(self):
                super(net, self).__init__()

                # build the feed-forward network -> sectors, edge, patch-relative & time-memory
                self.feedforward = make_simple_feedforward(**feedforward)

                # build the final stage
                self.final_stage = make_simple_feedforward(**final_stage)
                
                # for action advantage estimates
                self.valueL = nn.Linear(final_stage['linearsDim'][-1], 1)
                self.actadvs = nn.Linear(final_stage['linearsDim'][-1], outDims)

                # indices to split at
                self.f_part = (0, 4*num_sectors+7)


            def forward(self, input:Tensor):

                # split and reshape input
                if debug: print(input.shape)
                feedforward_part = input[:,self.f_part[0]:self.f_part[1]]

                # process feedforward_part
                if debug: print('\nfeedforward_part',feedforward_part.shape)
                for layer in self.feedforward: feedforward_part = layer(feedforward_part)         

                # process merged features
                if debug: print('concat',feedforward_part.shape)
                for layer in self.final_stage: feedforward_part = layer(feedforward_part)

                value = self.valueL(feedforward_part)
                advs = self.actadvs(feedforward_part)
                qvalues = value + (advs - advs.mean())
                
                return qvalues

        # # show the architecture (by backward pass fn)
        # if saveVizpath:
        #     head = os.path.split(saveVizpath)[0]
        #     if not os.path.exists(head): os.makedirs(head)
        #     nnet = net()
        #     viz= torchviz.make_dot(nnet(
        #         torch.tensor([self.makeState([[None]*4],None)], dtype=torch.float32)
        #     ), params=dict(list(nnet.named_parameters())))
        #     viz.render(saveVizpath, format='png')
        #     del nnet

        if debug:
            net().to(TORCH_DEVICE)(
                torch.tensor([self.makeState([[None]*4],None)], 
                    dtype=torch.float32, device=TORCH_DEVICE))

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

        fig, ax = plt.subplots(2,2, figsize=(15, 10))
        plt.tight_layout(pad=5)
        staterecord = open(os.path.join(debugDir, 'stMakerdebugstate.txt'), 'r')
        envrecord = open(os.path.join(debugDir, 'stMakerrecordenv.txt'), 'r')
        action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'EX']
        
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
            patch_relative = state[4*self.num_sectors+4]
            total_juice = state[4*self.num_sectors+4+1]
            time_mem = state[4*self.num_sectors+4+2]

            berries = agent_and_berries[1:]
            agent = agent_and_berries[0]
            w,h = self.berryField.OBSERVATION_SPACE_SIZE
            W, H = self.berryField.FIELD_SIZE

            ax[0][0].imshow(sectorized_states)
            ax[0][1].bar([0,1,2],[total_juice, patch_relative, time_mem], [1,1,1])
            ax[1][0].bar([*range(4)],edge_dist)

            # draw the berry-field
            ax[1][1].add_patch(Rectangle((agent[0]-w/2, agent[1]-h/2), w,h, fill=False))
            ax[1][1].add_patch(Rectangle((agent[0]-w/2-30,agent[1]-h/2-30), w+60,h+60, fill=False))
            ax[1][1].scatter(x=berries[:,0], y=berries[:,1], s=berries[:,2], c='r')
            ax[1][1].scatter(x=agent[0], y=agent[1], s=agent[2], c='black')
            if agent[0]-w/2 < 0: ax[1][1].add_patch(Rectangle((0, agent[1] - h/2), 1, h, color='blue'))
            if agent[1]-h/2 < 0: ax[1][1].add_patch(Rectangle((agent[0] - w/2, 0), w, 1, color='blue'))
            if W-agent[0]-w/2<0: ax[1][1].add_patch(Rectangle((W, agent[1] - h/2), 1, h, color='blue'))
            if H-agent[1]-h/2<0: ax[1][1].add_patch(Rectangle((agent[0] - w/2, H), w, 1, color='blue'))

            # compute q-values and plot qvals
            if nnet: 
                originalqvals = nnet(torch.tensor([state], dtype=torch.float32)).detach()[0].numpy()
                maxidx = np.argmax(originalqvals)
                ax[1][1].text(agent[0]+20, agent[1]+20, f'q:{originalqvals[maxidx]:.2f}:{action_names[maxidx]}')

                # add action-advs circles
                colorqs = originalqvals[:8]
                colors = (colorqs-min(colorqs))/(max(colorqs)-min(colorqs)+EPSILON)
                for angle in range(0, 360, self.angle):
                    rad = 2*np.pi * (angle/360)
                    x,y = 100*np.sin(rad), 100*np.cos(rad)
                    c = colors[angle//self.angle]
                    ax[1][1].add_patch(Circle((agent[0]+x, agent[1]+y), 20, color=(c,c,0,1)))

                # set title
                str_qvals = [f"{np.round(x,2):.2f}" for x in originalqvals.tolist()]
                meanings = [action_names[i]+' '*(len(qv)-len(action_names[i])) for i,qv in enumerate(str_qvals)]
                ax[1][1].set_title(f'env-record with q-vals plot\nqvals: {" ".join(str_qvals)}\n       {" ".join(meanings)}')

            # titles and ticks
            ax[0][0].set_title('sectorized states')
            ax[0][1].set_title('measure of patch-center-dist')
            ax[0][1].set_xticklabels(["","","total-juice","","patch-rel","","time-mem"]) 
            ax[1][0].set_title('measure of dist-from-edge')
            ax[1][0].set_xticklabels(["","","left","right","top","bottom"]) 
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
