from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# make custom state using the env.step output
# ks = 0.0001 - the reward rate of env
# kd = 0.011473 - the drain-rate times half diagonal of obs-space
def get_make_state(angle = 45, kd=0.011473, ks=0.0001, avf = 0.1, noise_scale=0.01):

    print('angle:', angle, ', kd:', kd, ', ks:', ks, 
            ', avf:', avf, ', noise_scale:', noise_scale)

    num_sectors = 360//angle

    # accumulators - for some sort of continuity
    m1 = np.zeros(num_sectors) # max-worth of each sector
    m2 = np.zeros(num_sectors) # stores densities of each sector
    m3 = np.zeros(num_sectors) # indicates the sector with the max worthy berry
    m4 = np.zeros(num_sectors) # a mesure of distance to max worthy in each sector

    def make_state(list_raw_observations, list_infos):
        nonlocal m1 ,m2, m3, m4
        # list_raw_observation a list of observations
        # raw_observations [x,y,size]

        raw_observation = list_raw_observations[-1]
        info = list_infos[-1]

        a1,a2,a3,a4 = avf*m1,avf*m2,avf*m3,avf*m4

        if info is None: 
            edge_dist = np.ones(4)
        else:
            edge_dist = info['scaled_dist_from_edge']

        if len(raw_observation) > 0:
            sizes = raw_observation[:,2]
            dist = np.linalg.norm(raw_observation[:,:2], axis=1)

            try:
                assert all(dist>0)
            except:
                print('osngoseinapinpawn')
                print(raw_observation)
                print(list_raw_observations)
                assert 0 > 1

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

                    # density of sector
                    density = np.sum(_sizes**2)/(1920*1080)
                    a2[idx] = density*100

                    # max worthy
                    worthinesses= ks*_sizes-kd*_dists
                    maxworthyness_idx = np.argmax(worthinesses)
                    a4[idx] = 1 - _dists[maxworthyness_idx]
                    worthyness = worthinesses[maxworthyness_idx]
                    a1[idx] = worthyness
                    if worthyness > maxworth:
                        maxworth_idx = idx
                        maxworth = worthyness             
            
            if maxworth_idx > -1: a3[maxworth_idx]=1 
            
        # make final state
        state = np.concatenate([a1,a2,a3,a4,edge_dist])

        # update accumulators
        m1,m2,m3,m4 = a1,a2,a3,a4

        # add noise
        state = state + np.random.randn(len(state))*noise_scale

        # print(state)
        return state

    return 4*num_sectors+4, make_state