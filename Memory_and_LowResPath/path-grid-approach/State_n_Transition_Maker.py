# remember the approx loc of unpicked berries
# remember a low-res path

from berry_field.envs.berry_field_env import BerryFieldEnv
from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

ROOT_2_INV = 0.5**(0.5)
EPSILON = 1E-8

class State_n_Transition_Maker():
    """ states containing an approximantion of the path of the agent 
    and also computed uisng info from all seen but not-collected berries """
    def __init__(self, berryField:BerryFieldEnv, mode='train', field_grid_size=(25,25), 
                    angle = 45, worth_offset=0.0, positive_emphasis=True) -> None:
        """ mode is required to assert whether it is required to make transitions """
        self.istraining = mode == 'train'
        self.angle = angle
        self.num_sectors = 360//angle        
        self.worth_offset = worth_offset
        self.berryField = berryField
        self.field_grid_size = field_grid_size
        self.positive_emphasis = positive_emphasis

        # init memories and other stuff
        self._init_memories()
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

    def _init_memories(self):
        memory_grid_size = self.field_grid_size[0]*self.field_grid_size[1]
        self.path_memory = np.zeros(memory_grid_size) # aprox path
        self.berry_memory = np.zeros(memory_grid_size) # aprox place of sighting a berry


    # for computation of berry worth, can help to change 
    # the agent's preference of different sizes of berries. 
    def berry_worth_function(self, sizes, distances):
        """ the reward that can be gained by pursuing a berry of given size and distance"""
        rr, dr = self.berryField.REWARD_RATE, self.berryField.DRAIN_RATE
        worth = rr * sizes - dr * distances + self.worth_offset
        worth = np.clip(worth, -1, 1)
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
        
        return [a1,a2,a3,a4], total_worth


    def _update_memories(self, info, total_worth):
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
        self.path_memory *= 0.999
        self.path_memory[index] = 1
        
        # update berry memory
        self.berry_memory[index] = 0.5*total_worth + 0.5*self.berry_memory[index]


    def computeState(self, raw_observation, info, reward, done) -> np.ndarray:
        """ makes a state from the observation and info. reward, done are ignored """
        # if this is the first state (a call to BerryFieldEnv.reset) -> marks new episode
        if info is None: # reinit memory and get the info and raw observation from berryField
            self._init_memories()
            raw_observation = self.berryField.raw_observation()
            info = self.berryField.get_info()

        # the total-worth is also representative of the percived goodness of observation
        sectorized_states, total_worth = self._compute_sectorized(raw_observation, info)

        # update memories
        self._update_memories(info, total_worth)

        # other extra information
        edge_dist = info['scaled_dist_from_edge']
        patch_relative = info['patch-relative']

        # make the state by concatenating sectorized_states and memories
        state = np.concatenate([
            *sectorized_states, edge_dist, patch_relative,
            self.berry_memory, self.path_memory
        ])

        return state


    def get_output_shape(self):
        if not self.output_shape:
            self.output_shape = self.computeState(None, None, None, None).shape
        return self.output_shape

    
    def makeState(self, skip_trajectory, action_taken):
        """ skip trajectory is a sequence of [[next-observation, info, reward, done],...] """
        if not self.istraining: return self.computeState(*skip_trajectory[-1])
        
        # if mode = 'train' we make the state-transitions (s,a,r,ns,d) for replay-buffer
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
                    
        return current_state
        
    
    def makeTransitions(self, skip_trajectory, state, action, nextState):
        """ get the state-transitions """
        return self.state_transitions
        

# CODE SNIPPETS

# if self.positive_emphasis:# more emphasis on positive rewards
#     for i in range(len(self.state_transitions)):
#         # if a berry was encountered
#         if self.state_transitions[i][2] > 0 and i > 0:
#             good_state = self.state_transitions[i][3]
#             reward_budget = self.state_transitions[i][2]

#             # go back till the reward becomes 0 (due to living expense)
#             sum_rewards = []
#             for j in range(i-1, -1, -1):
#                 reward_budget += self.state_transitions[j][2]
#                 sum_rewards.append(reward_budget)
#                 if reward_budget <= 0: break

#             # make the next_state as good_state (goal) for all k: j <= k < i
#             for k in range(j,i): 
#                 self.state_transitions[k][3] = good_state
#                 self.state_transitions[k][2] = sum_rewards[j-k]