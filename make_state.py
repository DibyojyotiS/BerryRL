from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# traits like selecting big or small berries, closest or 
# somewhere in between can be modeled by a preference 
# score 'worthiness' as ks*berry_size - kd*distance

# make custom state using the env.step output
# ks = 0.0001 - the reward rate of env
# kd = 0.011473 - the drain-rate times half diagonal of obs-space
def get_make_state(angle = 45, kd=0.011473, ks=0.0001, avf = 0.1, noise_scale=0.01):

    print('angle:', angle, ', kd:', kd, ', ks:', ks, 
            ', avf:', avf, ', noise_scale:', noise_scale)

    num_sectors = 360//angle

    # accumulators - for some sort of continuity
    m1 = np.zeros(num_sectors) # max-worth of each sector (transformed to 0-1 range, independently for each state)
    m2 = np.zeros(num_sectors) # stores densities of each sector (transformed to 0-1 range, independently for each state)
    m3 = np.zeros(num_sectors) # indicates the sector with the max worthy berry
    m4 = np.zeros(num_sectors) # a mesure of distance to max worthy in each sector

    def scale_01(vec,mask):
        tmp = vec*mask
        vec = vec - tmp
        tmp -= tmp.min()
        tmp /= (tmp.max()+1e-8)
        vec += tmp 
        return vec

    def make_state(trajectory):
        nonlocal m1 ,m2, m3, m4
        # trajectory is a list of [observation, info, reward, done]
        # raw_observations [x,y,size]

        raw_observation, info, reward, done = trajectory[-1]

        a1,a2,a3,a4 = avf*m1,avf*m2,avf*m3,avf*m4
 
        edge_dist=np.ones(4) if info is None else info['scaled_dist_from_edge']

        if len(raw_observation) > 0:
            hasberry = np.zeros(num_sectors)
            sizes = raw_observation[:,2]
            dist = np.linalg.norm(raw_observation[:,:2], axis=1)
            directions = raw_observation[:,:2]/dist[:,None]
            angles = getTrueAngles(directions)
            
            maxworth = float('-inf')
            maxworth_idx = -1
            for x in range(0,360,angle):
                sectorL = (x-angle/2)%360
                sectorR = (x+angle/2)
                if sectorL < sectorR:
                    args = np.argwhere((angles>=sectorL)&(angles<=sectorR))
                else:
                    args = np.argwhere((angles>=sectorL)|(angles<=sectorR))
                
                if args.shape[0] > 0: 
                    idx = x//angle
                    _sizes = sizes[args]
                    _dists = dist[args]
                    hasberry[idx] = 1

                    # density of sector
                    density = np.sum(_sizes**2)/(1920*1080)
                    a2[idx] = density*100

                    # max worthy
                    worthinesses= ks*_sizes-kd*_dists
                    maxworthyness_idx = np.argmax(worthinesses)
                    a4[idx] = 1 - _dists[maxworthyness_idx]
                    a1[idx] = worthyness = worthinesses[maxworthyness_idx]
                    if worthyness > maxworth:
                        maxworth_idx = idx
                        maxworth = worthyness             
            
            if maxworth_idx > -1: a3[maxworth_idx]=1 

            # 0-1 normalize max-worths(a1) for sectors with berries
            a1 = scale_01(a1, hasberry)

            # 0-1 normalize densities(a2) for sectors with berries
            a2 = scale_01(a2, hasberry)
            
        # make final state
        state = np.concatenate([a1,a2,a3,a4,edge_dist])

        # update accumulators
        m1,m2,m3,m4 = a1,a2,a3,a4

        # add noise - sometimes jolts out a stuck agent
        state = state + np.random.randn(len(state))*noise_scale

        # print(state)
        return state

    return 4*num_sectors+4, make_state



def get_make_transitions(make_state_fn, look_back=100):
    # lookback set to a high value to practically give s(s-1) transitions

    def make_transitions(trajectory, state, action, nextState):
        # trajectory is a list of [observation, info, reward, done]
        # returns transitions as [[state, action, reward, next-state, done],...]

        rewards = np.cumsum([r for o,i,r,d in trajectory])
        transitions = []
        for i in range(len(trajectory)):
            if trajectory[i][2] <= 0: continue
            berry_hit_state = make_state_fn([trajectory[i]])
            for j in range(max(0,i-look_back),i):
                transitions.append([
                    make_state_fn([trajectory[j]]), # state
                    action,
                    rewards[i]-rewards[j],
                    berry_hit_state,
                    False
                ])

        done = trajectory[-1][3]
        transitions.append([state, action, rewards[-1], nextState, done])
        return transitions

    return make_transitions