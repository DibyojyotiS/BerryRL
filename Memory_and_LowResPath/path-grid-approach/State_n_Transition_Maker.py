# remember the approx loc of unpicked berries
# remember a low-res path

import os
from berry_field.envs.berry_field_env import BerryFieldEnv
from berry_field.envs.utils.misc import getTrueAngles
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT_2_INV = 0.5**(0.5)
EPSILON = 1E-8

class State_n_Transition_Maker():
    """ states containing an approximantion of the path of the agent 
    and also computed uisng info from all seen but not-collected berries """
    def __init__(self, berryField:BerryFieldEnv, mode='train', field_grid_size=(25,25), 
                    angle = 45, worth_offset=0.0, noise=0.05, positive_emphasis=True,
                    memory_alpha=0.99, debug=False, debugDir='.temp/stMakerdebug') -> None:
        """ mode is required to assert whether it is required to make transitions """
        self.istraining = mode == 'train'
        self.angle = angle
        self.worth_offset = worth_offset
        self.berryField = berryField
        self.field_grid_size = field_grid_size
        self.positive_emphasis = positive_emphasis
        self.noise = noise
        self.memory_alpha = memory_alpha

        # init memories and other stuff
        self._init_memories()
        self.num_sectors = 360//angle        
        self.state_transitions = []
        self.output_shape = None

        self.divLenX = berryField.FIELD_SIZE[0]//field_grid_size[0]
        self.divLenY = berryField.FIELD_SIZE[1]//field_grid_size[1]

        if self.positive_emphasis:
            print('''positive rewards are now emphasised in the state-transitions
            Once a berry is encountered (say at index i), new transitions of the following
            description will also be appended: all the transitions k < i
            such that the sum of reward from k to i is positive will have the 
            next-state replaced by the state at transition at index i. And the rewards
            will also be replaced by the summation from k to i.\n''')
        
        # setup debug
        self.debugDir = debugDir
        self.debug = debug
        if debug:
            if not os.path.exists(debugDir): os.makedirs(debugDir)
            self.state_debugfile = open(os.path.join(debugDir, 'stMakerdebugstate.txt'), 'w', 1)
            self.env_recordfile = open(os.path.join(debugDir, 'stMakerrecordenv.txt'), 'w', 1)


    def _init_memories(self):
        memory_grid_size = self.field_grid_size[0]*self.field_grid_size[1]
        self.path_memory = np.zeros(memory_grid_size) # aprox path
        self.berry_memory = np.zeros(memory_grid_size) # aprox place of sighting a berry

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
        a1 = np.zeros(self.num_sectors) # max-worth of each sector
        a2 = np.zeros(self.num_sectors) # stores worth-densities of each sector
        a3 = np.zeros(self.num_sectors) # indicates the sector with the max worthy berry
        a4 = np.zeros(self.num_sectors) # a mesure of distance to max worthy in each sector
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
                    a2[idx] = np.sum(worthinesses)/10
                    a4[idx] = 1 - _dists[maxworthyness_idx]
                    total_worth += sum(worthinesses)
                    if worthyness > maxworth:
                        maxworth_idx = idx
                        maxworth = worthyness    
            if maxworth_idx > -1: a3[maxworth_idx]=1 
        
        
        avg_worth = total_worth/len(raw_observation) if len(raw_observation) > 0 else 0

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


    def computeState(self, raw_observation, info, reward, done) -> np.ndarray:
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
        state = np.concatenate([
            *sectorized_states, edge_dist, patch_relative,
            self.berry_memory, self.path_memory
        ])

        return state + np.random.uniform(-self.noise, self.noise, size=state.shape)

    
    def _compute_transitons_n_finalstate(self, skip_trajectory, action_taken):
        self.state_transitions = []
        current_state = self.computeState(*skip_trajectory[0])
        for i in range(1, len(skip_trajectory)):
            reward, done = skip_trajectory[i][2:]
            next_state = self.computeState(*skip_trajectory[i])
            self.state_transitions.append([current_state, action_taken, reward, next_state, done])
            current_state = next_state
        
        if self.positive_emphasis:# more emphasis on positive rewards
            # find where berries were encountered
            berry_indices = [0] + [i for i,x in enumerate(self.state_transitions) if x[2] > 0]
            # for each k: a < k < b for consequitive a,b in berry indices
            # append transitions with start-state at k and next-state at b (goal)
            # if the summed reward from k to b is positive
            for i in range(1, len(berry_indices)):
                a, b = berry_indices[i-1:i+1]
                if a+1 >= b: continue
                reward_b = self.state_transitions[b][2]
                good_state = self.state_transitions[b][3]
                for k in range(b-1, a, -1):
                    s, a, r, ns, d = self.state_transitions[k]
                    reward_b += r
                    self.state_transitions.append([s,a,reward_b,good_state,d])
                    if reward_b <= 0: break
        
        return current_state # the final state

    def get_output_shape(self):
        if not self.output_shape:
            self.output_shape = self.computeState(None, None, None, None).shape
        return self.output_shape

    def makeState(self, skip_trajectory, action_taken):
        """ skip trajectory is a sequence of [[next-observation, info, reward, done],...] """
        if not self.istraining: 
            final_state = self.computeState(*skip_trajectory[-1])
        else:
            final_state = self._compute_transitons_n_finalstate(skip_trajectory, action_taken)

        # debug
        if self.debug: 
            agent, berries = self.berryField.get_human_observation()
            np.savetxt(self.state_debugfile, [final_state])
            np.savetxt(self.env_recordfile, [np.concatenate([agent, *berries[:,:3]])])

        return final_state
        
    
    def makeTransitions(self, skip_trajectory, state, action, nextState):
        """ get the state-transitions """
        return self.state_transitions


    def showDebug(self, debugDir = None):
        
        # close the log files if not already closed
        if self.debug and not self.state_debugfile.closed:
            self.state_debugfile.write('end')
            self.state_debugfile.close()
        if self.debug and not self.env_recordfile.closed:
            self.env_recordfile.write('end')
            self.env_recordfile.close()
        
        if not debugDir: debugDir = self.debugDir

        fig, ax = plt.subplots(2,3, figsize=(15, 10))
        plt.tight_layout()
        f = open(os.path.join(debugDir, 'stMakerdebugstate.txt'), 'r')
        g = open(os.path.join(debugDir, 'stMakerrecordenv.txt'), 'r')
        while True:
            line = f.readline()
            line2 = g.readline()
            if line == 'end': break

            state = np.array(eval('[' + line[:-1].replace(' ', ',') + ']'), dtype=float)
            agent_and_berries = np.array(eval('[' + line2[:-1].replace(' ', ',') + ']'), dtype=float).reshape(-1,3)

            sectorized_states = state[:4*self.num_sectors].reshape(4,self.num_sectors)
            edge_dist = state[4*self.num_sectors: 4*self.num_sectors+4]
            patch_relative = state[4*self.num_sectors+4:4*self.num_sectors+4+1]
            memories = state[4*self.num_sectors+4+1:]
            berry_memory = memories[:self.field_grid_size[0]*self.field_grid_size[1]].reshape(self.field_grid_size)
            path_memory = memories[self.field_grid_size[0]*self.field_grid_size[1]:].reshape(self.field_grid_size)

            berries = agent_and_berries[1:]
            agent = agent_and_berries[0]
            w,h = self.berryField.OBSERVATION_SPACE_SIZE
            W, H = self.berryField.FIELD_SIZE

            ax[0][0].imshow(sectorized_states)
            ax[0][1].bar([1,2],[1,patch_relative], [0,1])
            ax[0][2].bar([*range(4)],edge_dist)
            ax[1][0].imshow(berry_memory)
            ax[1][1].imshow(path_memory)

            # draw the berry-field
            ax[1][2].scatter(x=berries[:,0], y=berries[:,1], s=berries[:,2], c='r')
            ax[1][2].scatter(x=agent[0], y=agent[1], s=agent[2], c='black')
            ax[1][2].add_patch(Rectangle((agent[0]-w/2, agent[1]-h/2), w,h, fill=False))
            ax[1][2].add_patch(Rectangle((agent[0]-w/2-30,agent[1]-h/2-30), w+60,h+60, fill=False))
            if agent[0]-w/2 < 0: ax[1][2].add_patch(Rectangle((0, agent[1] - h/2), 1, h, color='blue'))
            if agent[1]-h/2 < 0: ax[1][2].add_patch(Rectangle((agent[0] - w/2, 0), w, 1, color='blue'))
            if W-agent[0]-w/2<0: ax[1][2].add_patch(Rectangle((W, agent[1] - h/2), 1, h, color='blue'))
            if H-agent[1]-h/2<0: ax[1][2].add_patch(Rectangle((agent[0] - w/2, H), w, 1, color='blue'))

            plt.pause(0.001)
            
            for b in ax: 
                for a in b: a.clear() 
            
        plt.show()
        plt.close()

        f.close()
        g.close()