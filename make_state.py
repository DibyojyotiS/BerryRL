from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# make custom state using the env.step output
def get_make_state(angle = 45, kd=0.1, ks=0.5):
    num_sectors = 360//angle
    m1 = np.zeros(num_sectors)
    m2 = np.zeros(num_sectors)
    m3 = np.zeros(num_sectors)

    def make_state(list_raw_observations, list_infos):
        nonlocal m3, m2, m1
        # list_raw_observation a list of observations
        # raw_observations [x,y,size]

        raw_observation = list_raw_observations[-1]
        info = list_infos[-1]

        if info is None: 
            edge_dist = np.ones(4)
        else:
            edge_dist = info['scaled_dist_from_edge']

        sizes = raw_observation[:,2]
        dist = np.linalg.norm(raw_observation[:,:2], axis=1, keepdims=True)
        directions = raw_observation[:,:2]/dist
        angles = getTrueAngles(directions)

        a1 = np.zeros(num_sectors) # indicates sector with closest berry
        a2 = np.zeros_like(a1) # stores densities of each sector
        a3 = np.zeros_like(a1) # indicates the sector with the max worthy berry
        a4 = np.zeros_like(a1) # distance to closest berry in each sector
        
        maxworth = float('-inf')
        maxworth_idx = -1
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
                a2[idx] = density
                # closest dist
                current_closest = np.min(dist[args])
                a4[idx] = current_closest
                if current_closest < closest_dist:
                    closest_dist = current_closest
                    closest_idx = idx
                # max worthy
                worthyness = np.max(ks*_sizes-kd*_dists)
                if worthyness > maxworth:
                    maxworth_idx = idx
                    maxworth = worthyness
        
        a1,a3 = 0.5*m1,0.5*m3
        if maxworth_idx > -1: a3[maxworth_idx]=1 
        if closest_idx > -1: a1[closest_idx] = 1
        a2 = a2/(np.max(a2)+1E-6)
        m1,m3 = a1,a3
        
        # make final state
        state = np.concatenate([a1,a2,a3,a4,edge_dist])
        # print(state)
        return state

    return 4*num_sectors+4, make_state