from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# traits like selecting big or small berries, closest or 
# somewhere in between can be modeled by a preference 
# score 'worthiness' as ks*berry_size - kd*distance

# make custom state using the env.step output
# 0.0001 - the reward rate of env
# 0.011473 - the drain-rate times half diagonal of obs-space
def get_make_state(angle = 45, avf = 0.8, noise_scale=0.01, kd=0.011473, ks=0.0001, offset=0.01, BerrySizeRange=[10,50]):

    print('angle:', angle, ', kd:', kd, ', ks:', ks, ', avf:', avf, 
            ', offset:',offset, ', noise_scale:', noise_scale)
        
    # for computation of berry worth, can help to change 
    # the agent's preference of different sizes of berries. 
    minBerrySize, maxBerrySize = BerrySizeRange
    min_val = ks*minBerrySize - kd
    max_val = (ks*maxBerrySize - min_val)/(1-offset)
    ks/=max_val; kd/=max_val; min_val/=max_val
    worth_function = lambda size,dist: ks*size-kd*dist-min_val+offset

    num_sectors = 360//angle
    # traces - for some sort of continuity
    m1 = np.zeros(num_sectors) # max-worth of each sector
    m2 = np.zeros(num_sectors) # stores densities of each sector
    m3 = np.zeros(num_sectors) # indicates the sector with the max worthy berry
    m4 = np.zeros(num_sectors) # a mesure of distance to max worthy in each sector

    ROOT_2_INV = 0.5**(0.5)
    def make_state(trajectory):
        nonlocal m1 ,m2, m3, m4
        # trajectory is a list of [observation, info, reward, done]
        # raw_observations [x,y,size]

        # trace of previous state
        a1,a2,a3,a4 = avf*m1,avf*m2,avf*m3,avf*m4

        raw_observation, info, reward, done = trajectory[-1]
        edge_dist=np.ones(4) if info is None else info['scaled_dist_from_edge']
        
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
                    # density of sector
                    density = np.sum(_sizes**2)/(1920*1080)
                    a2[idx] = density*100
                    # max worthy
                    worthinesses= worth_function(_sizes,_dists)
                    maxworthyness_idx = np.argmax(worthinesses)
                    a4[idx] = 1 - _dists[maxworthyness_idx]
                    a1[idx] = worthyness = worthinesses[maxworthyness_idx]
                    if worthyness > maxworth:
                        maxworth_idx = idx
                        maxworth = worthyness    

            if maxworth_idx > -1: a3[maxworth_idx]=1 
            
        # make final state & add noise - sometimes jolts out a stuck agent
        state = np.concatenate([a1,a2,a3,a4,edge_dist])
        state += np.random.randn(len(state))*noise_scale

        if done: 
            m1[:]=m2[:]=m3[:]=m4[:]=0 # reset memory when done
        else:
            m1,m2,m3,m4 = a1,a2,a3,a4 # update memories

        # print(state)
        return state

    return 4*num_sectors+4, make_state



def get_make_transitions(make_state, look_back=100, min_jump=5):
    # lookback set to a high value to practically give s(s-1) transitions

    # my bad, this requires its own make-state function since the above 
    # make-state is state-full
    def make_transitions(trajectory, state, action, nextState):
        # trajectory is a list of [observation, info, reward, done]
        # returns transitions as [[state, action, reward, next-state, done],...]
        # need to recover mk from state

        rewards = np.cumsum([r for o,i,r,d in trajectory])
        transitions = []
        for i in range(len(trajectory)):
            if trajectory[i][2] <= 0: continue
            berry_hit_state = make_state([trajectory[i]])
            for j in range(max(0,i-look_back),i):
                transitions.append([
                    make_state([trajectory[j]]), # state
                    action,
                    rewards[i]-rewards[j],
                    berry_hit_state,
                    False
                ])

        done = trajectory[-1][3]
        transitions.append([state, action, rewards[-1], nextState, done])
        return transitions

    return make_transitions