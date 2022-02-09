from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

# make custom state using the env.step output
a3_ = np.zeros(8)
def make_state(list_raw_observation, info, angle = 45, kd=0.4, ks=0.1):
    global a3_
    # list_raw_observation a list of observations
    # raw_observations [x,y,size]
    raw_observation = list_raw_observation[-1]
    sizes = raw_observation[:,2]
    dist = np.linalg.norm(raw_observation[:,:2], axis=1, keepdims=True)
    directions = raw_observation[:,:2]/dist
    angles = getTrueAngles(directions)

    a1 = np.zeros(360//angle) # indicates sector with berries
    a2 = np.zeros_like(a1) # stores densities of each sector
    a3 = np.zeros_like(a1) # indicates the sector with the max worthy berry
    
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
            a1[idx] = 1
            # density of sector
            density = np.sum(sizes[args]**2)/(1920*1080)
            a2[idx] = density
            # max worthy
            worthyness = np.max(ks*sizes[args]-kd*dist[args])
            if worthyness > maxworth:
                maxworth_idx = idx
                maxworth = worthyness
    if maxworth_idx > -1: a3[maxworth_idx]=1 
    
    a3 = 0.9*a3_ + 0.1*a3
    state = np.concatenate([a1,a2,a3])
    a3_ = a3
    return state