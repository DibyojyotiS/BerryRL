from matplotlib.axes import Axes
import numpy as np
from collections import deque
from berry_field.envs import BerryFieldEnv
from torch import Tensor, float32, nn, device, tensor
from agent_utils import (berry_worth, random_exploration_v2, 
    compute_sectorized, Debugging, PatchDiscoveryReward, 
    skip_steps, make_simple_feedforward, printLocals, plot_time_mem_curves)
from agent_utils.memories.multi_resolution_time_memory import MultiResolutionTimeMemory

ROOT_2_INV = 0.5**(0.5)
EPSILON = 1E-8

class Agent():
    """ states containing an approximantion of the path of the agent 
    and also computed uisng info from all seen but not-collected berries """
    def __init__(self, berryField:BerryFieldEnv,

                # params controlling the percepted reward, state and state-transitions
                angle = 45, persistence=0.8, worth_offset=0.05, 
                noise=0.01, nstep_transition=[1], positive_emphasis=0,
                skipStep=10, patch_discovery_reward=0.5, 
                add_exploration = True,
                reward_magnification = 1e4/25,
                perceptable_reward_range = [0,2],

                # params related to time memory
                time_memory_factor=0.6, time_memory_exp=1.0,
                time_memory_grid_sizes= [
                    (20,20),(50,50),(100,100),(200,200),(400,400)
                ],

                # params related to berry memory
                # berry_memory_grid_size = (400,400),

                # other params
                render=False, 
                debug=False, debugDir='.temp',
                device=device('cpu'),
                stats_enabled=True) -> None:
        """ 
        ### parameters
        - berryField: BerryFieldEnv instance

        #### params controlling the state and state-transitions
        - angle: int (default 45)
                - the observation space divided into angular sectors
                - number of sectors is 360/angle
        - persistence: float (default 0.8)
                - the persistence of vision for the sectorized states
                and the time-memory 
        - worth_offset: float (default 0.05)
                - the offset in the berry worth function
                - can be used as an indicator for the exisitence of a berry
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
        - skipStep: int (default 10)
                - action is repeated for next skipStep steps
        - patch_discovery_reward: float (default 0.5)
                - add patch_discovery_reward amount of reward on discovering a new patch
                - can be set to None to disable reward patch discovery
        - add_exploration: bool (default True)
                - adds exploration-subroutine as an action
        - reward_magnification: float
                - the actual env step reward is multiplied by this
                - the reward for patch discovery is seperate from
                env step and is not multiplied by this
        - perceptable_reward_range: tupple(int, int)
                - the min and max cap of the reward 

        #### params related to time memory
        - time_memory_factor: float (default 0.01)
                - increment the time of the current block
                by delta for each step in the block
                as an exponential average with (1-delta)
                where delta = time_memory_factor/resolution
                and resolution = berry-field-size/time_memory_sizes
        - time_memory_exp: float (default 1.0)
                - raise the stored time memory for the current block
                to time_memory_exp and feed to agent's state
        - time_memory_sizes: list[tuple[int,int]]
                - the berry-field is divided into (L,M) sized grid
                and the agent notes the time spent in each of the cell.
                The memory of the time spent in a particular cell gets 
                accessed when the agent is in that cell.
        
        #### params related to berry-memory
        - berry_memory_grid_size: tuple[int,int]
                - the berry-field is divided into (L,M) sized grid
                and we remember the average-size of the berries seen
                in a particular cell.
                - The average-size is estimated from the average worth 
                since the worth function is between 0 and 1.
                - Any cell with non-zero entry is accessable to the agent 
                at all times. Basically the agent sees a berry of average 
                size at the center of the cell.
                - we will remove a value/cell from memory when its average
                size falls below 10. Or the agent is too far.

        - render: bool (default False)
                - wether to render the agent 
         """
        printLocals('Agent', locals())
        self.angle = angle
        self.persistence = persistence
        self.worth_offset = worth_offset
        self.berryField = berryField
        self.noise = noise
        self.nstep_transitions = nstep_transition # for approx n-step TD effect
        self.patch_discovery_reward = patch_discovery_reward
        self.positive_emphasis = positive_emphasis
        self.add_exploration= add_exploration
        self.reward_magnification = reward_magnification
        self.perceptable_reward_range = perceptable_reward_range
        self.time_memory_factor = time_memory_factor
        self.time_memory_exp = time_memory_exp
        self.time_memory_grid_sizes = time_memory_grid_sizes 
        # self.berry_memory_grid_size = berry_memory_grid_size
        self.render = render
        self.skipSteps = skipStep
        self.device = device

        # init memories and other stuff
        self._init_memories()
        self.nnet = self.makeNet(TORCH_DEVICE=device)
        self.berryField.step = self.get_wrapped_env_step(self.berryField, render)

        # setup debug
        self.debugger = Debugging(debugDir=debugDir, 
            berryField=self.berryField) if debug else None

        # some sanity checks
        assert len(perceptable_reward_range) == 2

    def _init_memories(self):
        # for the approx n-step TD construct
        self.state_deque = deque(maxlen=max(self.nstep_transitions) + 1)
        self.state_deque.append([None,None,0]) # init deque

        # for persistence
        self.prev_sectorized_state = None

        # for time memory
        self.time_memory = MultiResolutionTimeMemory(
            time_memory_grid_sizes= self.time_memory_grid_sizes,
            berryField_FIELD_SIZE= self.berryField.FIELD_SIZE,
            time_memory_factor= self.time_memory_factor,
            time_memory_exp= self.time_memory_exp,
            persistence= self.persistence
        )

        # a different kind of berry-memory
        self.berry_memory = {}

        return

    def _reset_memories(self):
        self.time_memory.reset()
        self.state_deque.clear()
        self.state_deque.append([None,None,0]) # reinit deque
        self.prev_sectorized_state = None

    def _update_memories(self, info, avg_worth):
        x,y = info['position']
        if x == self.berryField.FIELD_SIZE[0]: x -= EPSILON
        if y == self.berryField.FIELD_SIZE[1]: y -= EPSILON

        # decay time memory and update time_memory
        self.time_memory.update(x,y)
        return

    def _berry_worth_func(self, sizes, dists):
        return berry_worth(sizes, dists, 
            REWARD_RATE=self.berryField.REWARD_RATE, 
            DRAIN_RATE=self.berryField.DRAIN_RATE, 
            HALFDIAGOBS=self.berryField.HALFDIAGOBS, 
            WORTH_OFFSET=self.worth_offset,
            min_berry_size=10, max_berry_size=40)

    def _new_episode_started(self):
        """updates to agent state after an episode ends"""
        self._reset_memories()

    def _computeState(self, raw_observation, info, reward, done) -> np.ndarray:
        """ makes a state from the observation and info. reward, done are ignored """
        # if this is the first state (a call to BerryFieldEnv.reset) -> marks new episode
        if info is None: # reinit memory and get the info and raw observation from berryField
            self._new_episode_started()
            raw_observation = self.berryField.raw_observation()
            info = self.berryField.get_info()

        # # add the berry-memory to raw observation
        # cx,cy = info['position']
        # memory = [[x-cx,y-cy,s] for x,y,s in self.berry_memory.values()]
        # if len(memory) > 0: raw_observation = np.concatenate([raw_observation,memory])

        # the total-worth is also representative of the percived goodness of observation
        sectorized_states, avg_worth = compute_sectorized(
                raw_observation=raw_observation, 
                info=info, berry_worth_function=self._berry_worth_func,
                half_diagonalL= self.berryField.HALFDIAGOBS, 
                prev_sectorized_state=self.prev_sectorized_state, 
                persistence=self.persistence, angle=self.angle)
        self.prev_sectorized_state = sectorized_states

        # update memories
        self._update_memories(info, avg_worth)

        # other extra information
        edge_dist = info['scaled_dist_from_edge']
        patch_relative = info['patch-relative']
        total_juice = info['total_juice']
        any_berry_picked = len(info["recently_picked_berries"]) > 0

        # make the state by concatenating sectorized_states and memories
        state = np.concatenate([
            *sectorized_states, edge_dist, patch_relative, 
            [total_juice], self.time_memory.get_time_memories(),
            [any_berry_picked]
        ])

        return state + np.random.uniform(-self.noise, self.noise, size=state.shape)

    def get_wrapped_env_step(self, berryField:BerryFieldEnv, render=False):
        """ kinda magnifies rewards self.reward_scale for better gradients, 
        also rewards are clipped within self.perceptable_reward_range"""
        
        nactions = berryField.action_space.n
        berry_env_step = berryField.step
        reward_min, reward_max = self.perceptable_reward_range

        print(f'rewards scaled by {self.reward_magnification}')
        print(f'scaled rewards are clipped between {reward_min} and {reward_max}')

        # add the exploration subroutine to env action space
        if self.add_exploration:
            print('Exploration subroutine as an action')
            exploration_step = random_exploration_v2(berryField, 
                model=self.nnet, makeState=self.makeState, 
                hasInternalMemory=False, skipSteps=self.skipSteps,
                device=self.device, render=render)
            nactions+=1

        if self.patch_discovery_reward is not None and self.patch_discovery_reward != 0:
            print("Rewarding patch discovery")
            get_patch_discovery_reward = PatchDiscoveryReward(reward_value=self.patch_discovery_reward)

        # some stats to track; env based, so defined here so that they are 
        # independent for the training and the eval environment instances
        actual_steps = 0
        episode = 0
        action_counts = {i:0 for i in range(nactions)}

        def step(action):
            nonlocal actual_steps, episode

            # execute the action
            if self.add_exploration and action == nactions-1:
                reward, skip_trajectory, steps = exploration_step()
            else:
                if render: berryField.render()
                reward, skip_trajectory, steps = skip_steps(action=action, 
                    skipSteps= self.skipSteps, berryenv_step= berry_env_step)
            listOfBerries, info, _, done = skip_trajectory[-1]

            # update stats
            actual_steps += steps
            action_counts[action] += 1

            # modify reward
            scaled_reward = self.reward_magnification*reward
            if self.patch_discovery_reward is not None and self.patch_discovery_reward != 0: 
                scaled_reward += get_patch_discovery_reward(info)
            clipped_reward = min(max(scaled_reward, reward_min), reward_max)
            

            # print stuff and reset stats and patch-discovery-reward
            if done: 
                print(
                    f'\n=== episode:{episode} Env-steps-taken:{actual_steps}\n',
                    '\tactions counts:',action_counts,
                )
                actual_steps = 0; episode+=1
                if self.patch_discovery_reward is not None and self.patch_discovery_reward != 0: 
                    get_patch_discovery_reward(info=None)
                for k in action_counts:action_counts[k]=0
            
            return listOfBerries, clipped_reward, done, info
        return step

    def makeState(self, skip_trajectory, action_taken):
        """ skip trajectory is a sequence of [[next-observation, info, reward, done],...] """
        
        # the last info's recently_picked_berries must be a concat of all the recently_picked_berries
        berries_picked_in_trajectory = [
            l for x in skip_trajectory for l in (x[1]["recently_picked_berries"] if x[1] is not None else [])
        ]
        if skip_trajectory[-1][1] is not None:
            skip_trajectory[-1][1]["recently_picked_berries"] = berries_picked_in_trajectory

        final_state = self._computeState(*skip_trajectory[-1])
        if self.debugger: self.debugger.record(final_state)
        return final_state
      
    def makeStateTransitions(self, skip_trajectory, state, action, nextState):
        """ Makes the state-transitions using the given inputs and also makes
        n-step transitions according to the argument nstep_transitions in the 
        agent's init. The reward for the n-step transition is a simple summation
        of the rewards from the individual transitions. """
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
                transition = [oldstate, oldaction, sum_reward, nextState, done]
                if self.positive_emphasis and sum_reward > 0:
                    transitions.extend([transition]*self.positive_emphasis)
                else: transitions.append(transition)
        return transitions

    def makeNet(agent_self, TORCH_DEVICE):
        """ create and return the model (a duelling net)
        note: calling this multiple times will re-make the model"""
        outDims = agent_self.berryField.action_space.n
        n_sectorized = (1+len(agent_self.spacings))*4*(360//agent_self.angle)
        if agent_self.add_exploration: outDims+=1
        
        # define the layers
        feedforward = dict(
            infeatures=n_sectorized+len(agent_self.time_memory_grid_sizes)+4+2+1, 
            linearsDim = [32,16,16], lreluslope=0.1)
        
        class net(nn.Module):
            def __init__(self):
                super(net, self).__init__()

                # build the feed-forward network -> sectors, edge, patch-relative & time-memory
                self.feedforward = make_simple_feedforward(**feedforward)

                # for action advantage estimates
                self.valueL = nn.Linear(feedforward['linearsDim'][-1], 1)
                self.actadvs = nn.Linear(feedforward['linearsDim'][-1], outDims)

            def forward(self, feedforward_part:Tensor, debug=False):

                # split and reshape input
                if debug: print(feedforward_part.shape)

                # process feedforward_part
                if debug: print('\nfeedforward_part',feedforward_part.shape)
                for layer in self.feedforward: feedforward_part = layer(feedforward_part)         

                value = self.valueL(feedforward_part)
                advs = self.actadvs(feedforward_part)
                qvalues = value + (advs - advs.mean())

                return qvalues

        agent_self.nnet = net().to(TORCH_DEVICE)
        print('total-params: ', sum(p.numel() for p in agent_self.nnet.parameters() if p.requires_grad))
        return agent_self.nnet

    def getNet(self,debug=False)->nn.Module:
        """ return the agent's brain (a duelling net)"""
        if debug:
            self.nnet(tensor([self.makeState([[None]*4],None)],
                dtype=float32,device=self.device),debug=debug)
        return self.nnet

    def showDebug(self, gif=False, f=20, figsize=(15,10)):
        
        x,y = self.prev_sectorized_state.shape
        length = x*y
        actions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 
                    'W', 'NW', 'EX']

        def plotfn(axs:Axes, state:np.ndarray, *args):
            sectorized_states = state[:length].reshape(x,y)
            edge_dist = state[length: length+4]
            patch_relative = state[length+4]
            total_juice = state[length+5]
            time_mem = state[length+6:]
            axs[0][0].imshow(sectorized_states)
            axs[0][1].bar([*range(2+len(time_mem))],[total_juice, 
                    patch_relative, *time_mem], [1]*(2+len(time_mem)))
            axs[1][0].bar([*range(4)],edge_dist)
            axs[0][0].set_title('sectorized states')
            axs[0][1].set_title('measure of patch-center-dist')
            axs[0][1].set_xticklabels(["", "total-juice","patch-rel",
                *[f"time-mem-{x}" for x in self.time_memory_grid_sizes]]) 
            axs[1][0].set_title('measure of dist-from-edge')
            axs[1][0].set_xticklabels(["","","left","right","top","bottom"]) 
            axs[0][1].set_ylim(top=1)

        plotfns = [(2,2), plotfn]
        self.debugger.showDebug(plotfns=plotfns, nnet=self.nnet,
            device=self.device, f=f,action_names=actions, 
            gif=gif, figsize=figsize)
