from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# make custom state using the env.step output
def get_make_state(angle = 45, kd=0.1, ks=0.5):
    num_sectors = 360//angle
    avf = 0.8

    # accumulators - for some sort of continuity
    m1 = np.zeros(num_sectors) # indicates sector with closest berry
    m2 = np.zeros(num_sectors) # stores densities of each sector
    m3 = np.zeros(num_sectors) # indicates the sector with the max worthy berry
    m4 = np.zeros(num_sectors) # a mesure of distance to closest berry in each sector

    def make_state(list_raw_observations, list_infos):
        nonlocal m3, m2, m1, m4
        # list_raw_observation a list of observations
        # raw_observations [x,y,size]

        raw_observation = list_raw_observations[-1]
        info = list_infos[-1]

        if info is None: 
            edge_dist = np.ones(4)
        else:
            edge_dist = info['scaled_dist_from_edge']

        sizes = raw_observation[:,2]
        dist = np.linalg.norm(raw_observation[:,:2], axis=1)
        directions = raw_observation[:,:2]/dist[:,None]
        angles = getTrueAngles(directions)

        a1,a2,a3,a4 = avf*m1,avf*m2,avf*m3,avf*m4
        
        maxworth = float('-inf')
        maxworth_idx = -1
        maxworth_dist = 1
        closest_dist = float('inf')
        closest_idx = -1
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

                # closest dist
                current_closest = np.min(dist[args])
                if current_closest < closest_dist:
                    closest_dist = current_closest
                    closest_idx = idx
                a4[idx] = 1 - current_closest

                # max worthy
                worthinesses= ks*_sizes-kd*_dists
                maxworthyness_idx = np.argmax(worthinesses)
                worthyness = worthinesses[maxworthyness_idx]
                if worthyness > maxworth:
                    maxworth_idx = idx
                    maxworth = worthyness
                    maxworth_dist = 1 - _dists[maxworthyness_idx]
        
        if maxworth_idx > -1: a3[maxworth_idx]=1 
        if closest_idx > -1: a1[closest_idx] = 1

        
        # make final state
        state = np.concatenate([a1,a2,a3,a4,edge_dist,[np.squeeze(maxworth_dist)]])

        # update accumulators
        m1,m2,m3,m4 = a1,a2,a3,a4

        # print(state)
        return state

    return 4*num_sectors+4+1, make_state