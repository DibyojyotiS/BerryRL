from copy import deepcopy
from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# traits like selecting big or small berries, closest or 
# somewhere in between can be modeled by a preference 
# score 'worthiness' as ks*berry_size - kd*distance

# make custom state using the env.step output
# 0.0001 - the reward rate of env
# 0.011473 - the drain-rate times half diagonal of obs-space

def get_make_state(angle = 45, avf = 0.9, noise_scale=0.01, kd=0.011473, ks=0.0001, offset=0.01, BerrySizeRange=[10,50], look_back=100):

    print('angle:', angle, ', kd:', kd, ', ks:', ks, ', avf:', avf, 
            ', offset:',offset, ', noise_scale:', noise_scale)
    print('with action-trace in state')

    num_sectors = 360//angle
    ROOT_2_INV = 0.5**(0.5)
    EPSILON = 1E-8

    # for computation of berry worth, can help to change 
    # the agent's preference of different sizes of berries. 
    min_val = ks*BerrySizeRange[0] - kd
    max_val = (ks*BerrySizeRange[1] - min_val)/(1-offset)
    ks/=max_val; kd/=max_val; min_val/=max_val

    def worth_function(sizes, distances):
        return ks*sizes-kd*distances-min_val+offset

    def initial_state():
        # traces - for some sort of continuity
        # need to add indicators for relation to current patch
        # add sence of exploration distance
        s1 = np.zeros(num_sectors) # max-worth of each sector
        s2 = np.zeros(num_sectors) # stores worth-densities of each sector
        s3 = np.zeros(num_sectors) # indicates the sector with the max worthy berry
        s4 = np.zeros(num_sectors) # a mesure of distance to max worthy in each sector
        actions = np.zeros(9)      # previous action - there are 9 possible actions
        edge_dist = np.ones(4)     # a representative of distance from edges
        state = np.concatenate([s1,s2,s3,s4,actions,edge_dist])
        return state

    def recover_stuff(state):
        a1,a2,a3,a4 = state[0:4*num_sectors].reshape(4,num_sectors)
        actions = state[4*num_sectors:4*num_sectors+9]
        edge_dist = state[4*num_sectors+9:]
        return a1,a2,a3,a4,actions,edge_dist


    _state_ = initial_state()
    _prev_state_ = _state_
    def make_state(trajectory, action_taken):
        nonlocal _state_, _prev_state_
        # trajectory is a list of [observation, info, reward, done]
        # raw_observations [x,y,size]

        # store a copy of the previous state
        _prev_state_ = np.copy(_state_)

        # trace of previous state
        a1,a2,a3,a4,actions,edge_dist = recover_stuff(_state_*avf)
        raw_observation, info, reward, done = trajectory[-1]

        if info is not None: edge_dist[:] = info['scaled_dist_from_edge']
        if action_taken is not None: actions[action_taken] = 1

        if len(raw_observation) > 0:
            sizes = raw_observation[:,2]
            dist = np.linalg.norm(raw_observation[:,:2], axis=1) + EPSILON
            directions = raw_observation[:,:2]/dist[:,None]
            angles = getTrueAngles(directions)
            
            dist = ROOT_2_INV*dist # range in 0 to 1
            maxworth = float('-inf')
            maxworth_idx = -1
            for x in range(0,360,angle):
                sectorL, sectorR = (x-angle/2)%360, (x+angle/2)
                if sectorL < sectorR:
                    args = np.argwhere((angles>=sectorL)&(angles<=sectorR))
                else:
                    args = np.argwhere((angles>=sectorL)|(angles<=sectorR))
                
                if args.shape[0] > 0: 
                    idx = x//angle
                    _sizes = sizes[args]
                    _dists = dist[args]
                    # max worthy
                    worthinesses= worth_function(_sizes,_dists)
                    maxworthyness_idx = np.argmax(worthinesses)
                    a1[idx] = worthyness = worthinesses[maxworthyness_idx]
                    a2[idx] = np.sum(worthinesses)/10
                    a4[idx] = 1 - _dists[maxworthyness_idx]
                    if worthyness > maxworth:
                        maxworth_idx = idx
                        maxworth = worthyness    
            if maxworth_idx > -1: a3[maxworth_idx]=1 
            
        # make final state & add noise - sometimes jolts out a stuck agent
        _state_ = np.concatenate([a1,a2,a3,a4,actions,edge_dist])
        next_state = _state_ + np.random.randn(len(_state_))*noise_scale

        if done: _state_ = initial_state()
        return next_state


    def make_transitions(trajectory, state, action, nextState):
        # trajectory is a list of [observation, info, reward, done]
        # returns transitions as [[state, action, reward, next-state, done],...]
        # need to recover mk from state
        nonlocal _state_, _prev_state_

        state_copy = _state_
        prev_state_copy = _prev_state_

        rewards = np.cumsum([r for o,i,r,d in trajectory])
        transitions = []
        for i in range(len(trajectory)):
            if trajectory[i][2] <= 0: continue

            # set the state to one before calling 
            # make_state for this trajectory
            _state_ = np.copy(prev_state_copy)
            berry_hit_state = make_state([trajectory[i]], action)

            for j in range(max(0,i-look_back),i):
                if rewards[i]-rewards[j] < 0: continue
                _state_ = np.copy(prev_state_copy)
                transitions.append([
                    make_state([trajectory[j]], action), # state
                    action,
                    rewards[i]-rewards[j],
                    berry_hit_state,
                    False
                ])

        done = trajectory[-1][3]
        transitions.append([state, action, rewards[-1], nextState, done])

        _state_ = state_copy
        return transitions

    return _state_.shape[0], make_state, make_transitions



# def get_make_state(angle = 45, avf = 0.9, noise_scale=0.01, kd=0.011473, ks=0.0001, offset=0.01, BerrySizeRange=[10,50], look_back=100, stride=2):

#     print('angle:', angle, ', kd:', kd, ', ks:', ks, ', avf:', avf, 
#             ', offset:',offset, ', noise_scale:', noise_scale)

#     num_sectors = 360//angle
#     ROOT_2_INV = 0.5**(0.5)
#     MEM_LEN = 10

#     # for computation of berry worth, can help to change 
#     # the agent's preference of different sizes of berries. 
#     min_val = ks*BerrySizeRange[0] - kd
#     max_val = (ks*BerrySizeRange[1] - min_val)/(1-offset)
#     ks/=max_val; kd/=max_val; min_val/=max_val

#     def worth_function(sizes, distances):
#         return ks*sizes-kd*distances-min_val+offset

#     def default_state():
#         s1 = np.zeros(num_sectors) # max-worth of each sector
#         s2 = np.zeros(num_sectors) # stores worth-densities of each sector
#         s3 = np.zeros(num_sectors) # indicates the sector with the max worthy berry
#         s4 = np.zeros(num_sectors) # a mesure of distance to max worthy in each sector
#         actions = np.zeros(9)      # previous action - there are 9 possible actions
#         edge_dist = np.ones(4)     # a representative of distance from edges
#         return s1,s2,s3,s4,actions,edge_dist

#     def process_trajectory(trajectory, action_taken):
#         a1,a2,a3,a4,actions,edge_dist = default_state()
#         raw_observation, info, reward, done = trajectory[-1]

#         if info is not None: edge_dist[:] = info['scaled_dist_from_edge']
#         if action_taken is not None: actions[action_taken] = 1

#         if len(raw_observation) > 0:
#             sizes = raw_observation[:,2]
#             dist = np.linalg.norm(raw_observation[:,:2], axis=1)
#             directions = raw_observation[:,:2]/dist[:,None]
#             angles = getTrueAngles(directions)
            
#             dist = ROOT_2_INV*dist # range in 0 to 1
#             maxworth = float('-inf')
#             maxworth_idx = -1
#             for x in range(0,360,angle):
#                 sectorL, sectorR = (x-angle/2)%360, (x+angle/2)
#                 if sectorL < sectorR:
#                     args = np.argwhere((angles>=sectorL)&(angles<=sectorR))
#                 else:
#                     args = np.argwhere((angles>=sectorL)|(angles<=sectorR))
                
#                 if args.shape[0] > 0: 
#                     idx = x//angle
#                     _sizes = sizes[args]
#                     _dists = dist[args]
#                     # max worthy
#                     worthinesses= worth_function(_sizes,_dists)
#                     maxworthyness_idx = np.argmax(worthinesses)
#                     a1[idx] = worthyness = worthinesses[maxworthyness_idx]
#                     a2[idx] = np.sum(worthinesses)/10
#                     a4[idx] = 1 - _dists[maxworthyness_idx]
#                     if worthyness > maxworth:
#                         maxworth_idx = idx
#                         maxworth = worthyness    
#             if maxworth_idx > -1: a3[maxworth_idx]=1 
            
#         # make final state & add noise - sometimes jolts out a stuck agent
#         new_state = np.concatenate([a1,a2,a3,a4,actions,edge_dist])
#         new_state = new_state + np.random.randn(len(new_state))*noise_scale
#         return new_state


#     _state_ = [np.concatenate(default_state())]*MEM_LEN
#     def make_state(trajectory, action_taken):
#         nonlocal _state_
#         # trajectory is a list of [observation, info, reward, done]
#         # raw_observations [x,y,size]

#         # remove the oldest entry
#         _state_.pop(0)

#         # append the latest state
#         _state_.append(process_trajectory(trajectory, action_taken))
#         next_state = np.concatenate(_state_)

#         # if episode ended, reset the memory
#         if trajectory[-1][-1]: _state_ = [np.concatenate(default_state())]*MEM_LEN
#         print(_state_)
#         return next_state


#     def make_transitions(trajectory, state, action, nextState):
#         # trajectory is a list of [observation, info, reward, done]
#         # returns transitions as [[state, action, reward, next-state, done],...]
#         nonlocal _state_

#         rewards = np.cumsum([r for o,i,r,d in trajectory])
#         transitions = []
#         for i in range(len(trajectory)):
#             if trajectory[i][2] <= 0: continue

#             berry_hit_state = process_trajectory(trajectory, action)

#             for j in range(max(0,i-look_back),i,stride):
#                 in_situ_state = process_trajectory([trajectory[j]], action)
#                 transitions.append([
#                     np.concatenate(_state_[:-1]+[in_situ_state]),
#                     action,
#                     rewards[i]-rewards[j],
#                     np.concatenate(_state_[:-2]+[in_situ_state, berry_hit_state]),
#                     False
#                 ])

#         done = trajectory[-1][3]
#         transitions.append([state, action, rewards[-1], nextState, done])
#         return transitions
#     return np.concatenate(_state_).shape[0], make_state, make_transitions