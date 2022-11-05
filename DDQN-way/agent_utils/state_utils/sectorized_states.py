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

def compute_sectorized(raw_observation:np.ndarray, info:dict, 
                        berry_worth_function:'function', half_diagonalL:float,
                        prev_sectorized_state=None, 
                        persistence=0.8,angle=45):
    """ 
    ### parameters
    - raw_observation: ndarray
            - observation returned from BerryFieldEnv.step
            - a numpy array with rows as [x,y,berry-size]
    - info: dict[str,Any] 
            - info returned from BerryFieldEnv.step 
    - berry_worth_function: function
            - a function that takes in the sizes (array) 
            and distances (array) and returns a array
            denoting the worth of each berry. 
    - half_diagonalL: float
            - the length of the half diagonal of the observation window
            - used to normalize the distance of berry from agent
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
    if prev_sectorized_state is not None:
        a1,a2,a3,a4 = prev_sectorized_state * persistence
    else:
        num_sectors = 360//angle
        a1,a2,a3,a4 = np.zeros((4,num_sectors))
        # a1: max-worth of each sector
        # a2: stores avg-worth of each sector
        # a3: a mesure of distance to max worthy in each sector
        # a4: indicates the sector with the max worthy berry
    
    total_worth = 0
    if len(raw_observation) > 0:
        sizes = raw_observation[:,2]
        dist = np.linalg.norm(raw_observation[:,:2], axis=1) + EPSILON
        directions = raw_observation[:,:2]/dist[:,None]
        angles = getTrueAngles(directions)
        
        # dist = ROOT_2_INV*dist # range in 0 to 1 ## oops! this is a bug!!        
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
                worthinesses= berry_worth_function(_sizes,_dists)
                maxworthyness_idx = np.argmax(worthinesses) # max worthy
                a1[idx] = worthyness = worthinesses[maxworthyness_idx]
                a2[idx] = np.average(worthinesses)
                a3[idx] = 1 - _dists[maxworthyness_idx]/half_diagonalL
                total_worth += sum(worthinesses)
                
                if worthyness > maxworth:
                    maxworth_idx = idx
                    maxworth = worthyness   
        if maxworth_idx > -1: a4[maxworth_idx]=1

    avg_worth = total_worth/len(raw_observation) \
        if len(raw_observation) > 0 else 0
    return np.array([a1,a2,a3,a4]), avg_worth

def compute_distance_sectorized(raw_observation:np.ndarray, info:dict, 
    berry_worth_function:'function', spacings:list=[], prev_sectorized_state=None, 
    persistence=0.8, angle=45, observation_space_size=(1920,1080)):
    """ Segments the raw-observtion of berries by 
    distances according to the spacings given. Then
    computes the sectorized state for each of the
    segments using compute_sectorized function.

    ### parameters
    - raw_observation: ndarray
            - observation returned from BerryFieldEnv.step
            - a numpy array with rows as [x,y,berry-size]
    - info: dict[str,Any] 
            - info returned from BerryFieldEnv.step 
    - berry_worth_function: function
            - a function that takes in the sizes (array) 
            and distances (array) and returns a array
            denoting the worth of each berry. 
    - spacings:list (default empty list)
            - a list containg floats between 0 and 1 in
            increasing order. 
            say a and b are two floats in the list, then
            all the berries that are in distance >= a and 
            <= b are outputed.
            - if spacings is an empty list, then it is
            the same as calling compute_sectorized
    - prev_sectorized_state: ndarray (default None)
            - previously computed sectorized state
            to be used for persistence of vision
            - if None, then persistence has no effect
    - persistence: float (default 0.8)
            - the amount of information from the 
            prev_sectorized_state that is retained 

    ### returns
    - sectorized_state: ndarray
    - avg_worth: float, the average worth so seen
    """

    if len(spacings)==0:
        return compute_sectorized(raw_observation, info, 
            berry_worth_function=berry_worth_function, 
            prev_sectorized_state=prev_sectorized_state,
            persistence=persistence,
            angle=angle)

    W,H = observation_space_size
    half_diagonal= ((H**2+W**2)**0.5)/2
    w = np.abs(raw_observation[:,0]*half_diagonal/(W/2))
    h = np.abs(raw_observation[:,1]*half_diagonal/(H/2))

    indices = np.max([np.searchsorted(spacings, w),
            np.searchsorted(spacings, h)], axis=0)

    dist_obs = [[] for _ in range(len(spacings)+1)]
    for i_, i in enumerate(indices): dist_obs[i].append(i_)

    sectorized = []; avg_worth = 0
    for i, obs in enumerate(dist_obs):
        prev_sec = None if prev_sectorized_state is None \
                        else prev_sectorized_state[i*4:(i+1)*4] 
        sectorized_, avg_worth_ = compute_sectorized(
                raw_observation[obs], info, 
                berry_worth_function=berry_worth_function, 
                prev_sectorized_state=prev_sec,
                persistence=persistence,
                angle=angle)
        sectorized.append(sectorized_)
        avg_worth += avg_worth_*len(obs)

    if len(raw_observation) > 0: avg_worth/=len(raw_observation)
    sectorized = np.concatenate(sectorized, axis=0)
    return sectorized, avg_worth