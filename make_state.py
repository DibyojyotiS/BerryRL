from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# make custom state using the env.step output
m3 = np.zeros(8)
def make_state(list_raw_observation, info, angle = 45, kd=0.4, ks=0.1):
    global m3
    # list_raw_observation a list of observations
    # raw_observations [x,y,size]
    raw_observation = list_raw_observation[-1]
    sizes = raw_observation[:,2]
    dist = np.linalg.norm(raw_observation[:,:2], axis=1, keepdims=True)
    directions = raw_observation[:,:2]/dist
    angles = getTrueAngles(directions)

    a1 = np.zeros(360//angle) # indicates sector with closest berry
    a2 = np.zeros_like(a1) # stores densities of each sector
    a3 = np.zeros_like(a1) # indicates the sector with the max worthy berry
    
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
            if current_closest < closest_dist:
                closest_dist = current_closest
                closest_idx = idx
            # max worthy
            worthyness = np.max(ks*_sizes-kd*_dists)
            if worthyness > maxworth:
                maxworth_idx = idx
                maxworth = worthyness
    if maxworth_idx > -1: a3[maxworth_idx]=1 
    if closest_idx > -1: a1[closest_idx] = 1
    
    a3 = np.clip(0.5*m3 + a3, 0, 1)
    state = np.concatenate([a1,a2,a3])
    state = state + np.random.normal(scale=0.001,size=state.shape)
    m3 = a3
    # print(state)
    return state