import numpy as np
from berry_field.envs import BerryFieldEnv

EPSILON = 1e-8
ROOT_2_INV = 0.5**0.5

def getTrueAngles(directions, referenceVector=[0,1]):
    """ get the angle of the vectors given in the list 
    directions relative to the reference vector in [0,2π]
    - directions: ndarray
            - 2d array where the rows are treated as vectors
    - referenceVector: array | list
            - the reference vector from which the angles are
            measured in the clockwise sense
    returns the angles """
    curls = np.cross(directions, referenceVector)
    dot = np.dot(directions, referenceVector)
    angles = np.arccos(dot)*180/np.pi
    args0 = np.argwhere(np.bitwise_not((curls > 0)|(curls == 0)&(dot==1)))
    angles[args0] = 360-angles[args0]
    return angles

def compute_sectorized(raw_observation, info, 
                        berry_worth_function:'function', 
                        prev_sectorized_state=None, 
                        persistence=0.8,angle=45):
    """ 
    ### parameters
    - raw_observation: ndarray
            - observation returned from BerryFieldEnv.step
    - info: dict[str,Any] 
            - info returned from BerryFieldEnv.step 
    - berry_worth_function: function
            - a function that takes in the sizes (array) 
            and distances (array) and returns a array
            denoting the worth of each berry. 
    - prev_sectorized_state: ndarray (default None)
            - previously computed sectorized state
            to be used for persistence of vision
            - if None, then persistence has no effect
    - persistence: float (default 0.8)
            - the amount of information from the 
            prev_sectorized_state that is retained 
            
    ### returns
    - sectorized_state: ndarray
    - avg_worth: float, the average worth so seen"""

    # apply persistence if prev_sectorized_state is given
    if prev_sectorized_state:
        a1,a2,a3,a4 = prev_sectorized_state * persistence
    else:
        num_sectors = 360//angle
        a1 = np.zeros(num_sectors) # max-worth of each sector
        a2 = np.zeros(num_sectors) # stores avg-worth of each sector
        a3 = np.zeros(num_sectors) # indicates the sector with the max worthy berry
        a4 = np.zeros(num_sectors) # a mesure of distance to max worthy in each sector
    
    total_worth = 0
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
                worthinesses= berry_worth_function(_sizes,_dists)
                maxworthyness_idx = np.argmax(worthinesses)
                a1[idx] = worthyness = worthinesses[maxworthyness_idx]
                a2[idx] = np.average(worthinesses)
                a4[idx] = 1 - _dists[maxworthyness_idx]
                total_worth += sum(worthinesses)
                if worthyness > maxworth:
                    maxworth_idx = idx
                    maxworth = worthyness    
        if maxworth_idx > -1: a3[maxworth_idx]=1 
    
    avg_worth = total_worth/len(raw_observation) \
        if len(raw_observation) > 0 else 0
    return np.array([a1,a2,a3,a4]), avg_worth