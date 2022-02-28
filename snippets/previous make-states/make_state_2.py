from copy import deepcopy
from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# traits like selecting big or small berries, closest or 
# somewhere in between can be modeled by a preference 
# score 'worthiness' as ks*berry_size - kd*distance

# make custom state using the env.step output
# 0.0001 - the reward rate of env
# 0.011473 - the drain-rate times half diagonal of obs-space
def get_make_state(angle = 45, avf = 0.9, noise_scale=0.01, kd=0.011473, ks=0.0001, offset=0.01, BerrySizeRange=[10,50]):

    print('angle:', angle, ', kd:', kd, ', ks:', ks, ', avf:', avf, 
            ', offset:',offset, ', noise_scale:', noise_scale)

    num_sectors = 360//angle
    ROOT_2_INV = 0.5**(0.5)

    # for computation of berry worth, can help to change 
    # the agent's preference of different sizes of berries. 
    min_val = ks*BerrySizeRange[0] - kd
    max_val = (ks*BerrySizeRange[1] - min_val)/(1-offset)
    ks/=max_val; kd/=max_val; min_val/=max_val

    def worth_function(sizes, distances):
        return ks*sizes-kd*distances-min_val+offset

    def initial_state():
        # traces - for some sort of continuity
        s1 = np.zeros(num_sectors) # max-worth of each sector
        s2 = np.zeros(num_sectors) # stores worth-densities of each sector
        s3 = np.zeros(num_sectors) # indicates the sector with the max worthy berry
        s4 = np.zeros(num_sectors) # a mesure of distance to max worthy in each sector
        edge_dist = np.ones(4)     # a representative of distance from edges
        state = np.concatenate([s1,s2,s3,s4,edge_dist])
        return state

    def recover_stuff(state):
        a1,a2,a3,a4 = state[0:4*num_sectors].reshape(4,num_sectors)
        edge_dist = state[4*num_sectors:]
        return a1,a2,a3,a4,edge_dist


    # oversampled transitions
    _processed_trajectory_ = []

    _state_ = initial_state()
    def make_state(trajectory):
        nonlocal _state_
        # trajectory is a list of [observation, info, reward, done]
        # raw_observations [x,y,size]

        # clear the _processed_trajectory_
        _processed_trajectory_.clear()

        # trace of previous state
        a1,a2,a3,a4,edge_dist = recover_stuff(_state_)

        for i in range(len(trajectory)):

            a1,a2,a3,a4 = avf*a1,avf*a2,avf*a3,avf*a4
            raw_observation, info, reward, done = trajectory[i]

            if info is not None: edge_dist[:] = info['scaled_dist_from_edge']
            if len(raw_observation) > 0:
                sizes = raw_observation[:,2]
                dist = np.linalg.norm(raw_observation[:,:2], axis=1)
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
            _state_ = np.concatenate([a1,a2,a3,a4,edge_dist])
            next_state = _state_ + np.random.randn(len(_state_))*noise_scale

            # append state and reward to processed trajectory
            if reward != None: # reward is none for the first most trajectory
                _processed_trajectory_.append([next_state, reward, done])

        if done: _state_ = initial_state()
        # print(_state_)
        return next_state


    def make_transitions(trajectory, state, action, nextState):
        nonlocal _processed_trajectory_
        # trajectory is a list of [observation, info, reward, done]
        # returns transitions as [[state, action, reward, next-state, done],...]

        reward_cs = np.cumsum([r for ns,r,d in _processed_trajectory_])

        transitions = []
        last_hit_index = 0
        for i in range(1, len(_processed_trajectory_)):
            if _processed_trajectory_[i][1] <= 0: continue
            # else a berry is hit
            for j in range(last_hit_index+1, i):
                transitions.append([
                    _processed_trajectory_[j][0], # start-state
                    action,
                    reward_cs[i]-reward_cs[j],
                    _processed_trajectory_[i][0],
                    _processed_trajectory_[i][2] # berry-hit state
                ])
            last_hit_index = i

        transitions.append([state, action, reward_cs[-1], nextState, _processed_trajectory_[-1][2]])

        # erase _processed_trajectory
        _processed_trajectory_.clear()

        return transitions

    return 4*num_sectors+4, make_state, make_transitions