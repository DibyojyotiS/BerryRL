import numpy as np
from collections import deque
from berry_field.envs import BerryFieldEnv
from gym.spaces import Discrete
from torch import Tensor, nn, device
from agent_utils import berry_worth, compute_sectorized, Debugging, PatchDiscoveryReward
from utils import printLocals, random_exploration, make_simple_feedforward

ROOT_2_INV = 0.5**(0.5)
EPSILON = 1E-8

class Agent():
    """ states containing an approximantion of the path of the agent 
    and also computed uisng info from all seen but not-collected berries """
    def __init__(self, berryField:BerryFieldEnv, mode='train', 

                # params controlling the state and state-transitions
                angle = 45, persistence=0.8, worth_offset=0.0, 
                noise=0.01, nstep_transition=[1], positive_emphasis=0,
                reward_patch_discovery=True, 
                add_exploration = True,

                # params related to time memory
                time_memory_delta=0.01, time_memory_exp=1.0,

                # other params
                render=False, renderstep=10, 
                debug=False, debugDir='.temp',
                device=device('cpu')) -> None:
        """ 
        ### parameters
        - berryField: BerryFieldEnv instance
        - mode: 'train' or 'eval'

        #### params controlling the state and state-transitions
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

        #### params related to time memory
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
        self.device = device

        # init memories and other stuff
        self._init_memories()
        self.print_information()
        self.nnet = self.makeNet(TORCH_DEVICE=device, debug=debug)
        self.berryField.step = self.env_step_wrapper(self.berryField, render, renderstep)

        # setup debug
        self.debugger = Debugging(debugDir=debugDir, 
            OBSERVATION_SPACE_SIZE=self.berryField.OBSERVATION_SPACE_SIZE,
            BERRY_FIELD_SIZE=self.berryField.FIELD_SIZE) if debug else None

    def print_information(self):
        if not self.istraining: print('eval mode'); return
        if self.reward_patch_discovery: print("Rewarding patch discovery")
        if self.add_exploration: print("Exploration subroutine added")
        print('agent aware of total-juice')

    def env_step_wrapper(self, berryField:BerryFieldEnv, render=False, renderstep=10):
        """ kinda magnifies rewards by 2/(berry_env.REWARD_RATE*MAXSIZE)
        for better gradients..., also rewards are clipped between 0 and 2 """
        print('with living cost, rewards scaled by 2/(berryField.REWARD_RATE*MAXSIZE)')
        print('rewards are clipped between 0 and 2')
        
        MAXSIZE = max(berryField.berry_collision_tree.boxes[:,2])
        scale = 2/(berryField.REWARD_RATE*MAXSIZE)
        cur_n = berryField.action_space.n
        berry_env_step = berryField.step
        patch_discovery_reward = PatchDiscoveryReward(reward_value=1)

        # add the exploration subroutine to env action space
        if self.add_exploration:
            print('Exploration subroutine as an action')
            exploration_step = random_exploration(berryField, render=render, renderS=renderstep)
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
                reward += self.patch_discovery_reward(info)
            
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

        # for persistence
        self.prev_sectorized_state = None

        # for time memory
        self.time_memory = 0 # the time spent at the current block
        self.time_memory_data = np.zeros((200,200))
        return

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
        sectorized_states, avg_worth = compute_sectorized(raw_observation=raw_observation, 
                info=info, berry_worth_function=berry_worth)
        self.prev_sectorized_state = sectorized_states

        # update memories
        self._update_memories(info, avg_worth)

        # other extra information
        edge_dist = info['scaled_dist_from_edge']
        patch_relative = info['patch-relative']
        total_juice = info['total_juice']

        # make the state by concatenating sectorized_states and memories
        state = np.concatenate([*sectorized_states, edge_dist, patch_relative, 
                                [total_juice], [self.time_memory]])

        return state + np.random.uniform(-self.noise, self.noise, size=state.shape)

    def makeState(self, skip_trajectory, action_taken):
        """ skip trajectory is a sequence of [[next-observation, info, reward, done],...] """
        final_state = self._computeState(*skip_trajectory[-1])
        if self.debugger: self.debugger.record(final_state)
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
                transition = [oldstate, oldaction, sum_reward, nextState, done]
                if self.positive_emphasis and sum_reward > 0:
                    transitions.extend([transition]*self.positive_emphasis)
                else: transitions.append(transition)
        return transitions

    def makeNet(self, TORCH_DEVICE,
            feedforward = dict(infeatures=39, linearsDim = [32,16], lreluslope=0.1),
            final_stage = dict(infeatures=16, linearsDim = [8], 
                lreluslope=0.1)):
        """ create and return the model (a duelling net)
        note: calling this multiple times will re-make the model"""
        num_sectors = 360//self.angle
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

            def forward(self, input:Tensor, debug=False):

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

        self.nnet = net().to(TORCH_DEVICE)
        print('total-params: ', sum(p.numel() for p in self.nnet.parameters() if p.requires_grad))
        return self.nnet

    def getNet(self)->nn.Module:
        """ return the agent's brain (a duelling net)"""
        return self.nnet

    def showDebug(self, gif=False, f=20, figsize=(15,10)):
        
        num_sectors = 360//self.angle
        actions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 
                    'W', 'NW', 'EX']

        def plotfn(axs, state, *args):
            sectorized_states = state[:4*num_sectors].reshape(4,num_sectors)
            edge_dist = state[4*num_sectors: 4*num_sectors+4]
            patch_relative = state[4*num_sectors+4]
            total_juice = state[4*num_sectors+4+1]
            time_mem = state[4*num_sectors+4+2]
            axs[0][0].imshow(sectorized_states)
            axs[0][1].bar([0,1,2],[total_juice, patch_relative, time_mem], [1,1,1])
            axs[1][0].bar([*range(4)],edge_dist)
            axs[0][0].set_title('sectorized states')
            axs[0][1].set_title('measure of patch-center-dist')
            axs[0][1].set_xticklabels(["","","total-juice","","patch-rel","","time-mem"]) 
            axs[1][0].set_title('measure of dist-from-edge')
            axs[1][0].set_xticklabels(["","","left","right","top","bottom"]) 
            axs[0][1].set_ylim(top=1)

        plotfns = [(2,2), plotfn]
        self.debugger.showDebug(plotfns=plotfns, nnet=self.nnet,
            device=self.device, f=f,action_names=actions, 
            gif=gif, figsize=figsize)
