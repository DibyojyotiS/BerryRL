from typing import Callable, List, Tuple
import numpy as np
from numba import njit
from numpy import ndarray

EPSILON = 1e-8
ROOT_2_INV = 0.5**0.5


@njit
def njitGetTrueAngles(directions:ndarray, referenceVector=np.asfarray([0,1])):
    """ get the angle of the vectors given in the list 
    directions relative to the reference vector in [0,2π]
    - directions: ndarray
            - 2d array where the rows are treated as vectors
    - referenceVector: array | list
            - the reference vector from which the angles are
            measured in the clockwise sense
    returns the angles """
    curls = np.zeros(len(directions))
    for i in range(len(directions)):
        curls[i] = directions[i][0]*referenceVector[1] - directions[i][1] * referenceVector[0]
    
    dot = np.dot(directions, referenceVector)
    angles = np.arccos(dot)*180/np.pi
    args0 = np.nonzero(np.bitwise_not((curls > 0)|(curls == 0)&(dot==1)))[0]
    angles[args0] = 360-angles[args0]
    return angles


@njit
def njitSectorized(
    angles:ndarray, worths:ndarray, dist:ndarray, 
    array_out:ndarray,
    angle:int, maxPossibleDist:float
):
    a1,a2,a3,a4,a5,a6 = array_out
    for x in range(0,360,angle):
        sectorL = (x-angle/2)%360
        sectorR = (x+angle/2)
        if sectorL < sectorR:
            args = np.nonzero((angles>=sectorL)&(angles<=sectorR))[0]
        else:
            args = np.nonzero((angles>=sectorL)|(angles<=sectorR))[0]
        
        if args.shape[0] > 0: 
            idx = x//angle # sector
            sectorWorths:ndarray = worths[args]
            maxSecWorthIdx = args[np.argmax(sectorWorths)] # max worthy
            maxWorthDistInd = max(0, 1 - dist[maxSecWorthIdx]/maxPossibleDist)
            a1[idx] = worths[maxSecWorthIdx]
            a2[idx] = sectorWorths.mean()
            a3[idx] = maxWorthDistInd 
            a5[idx] = len(sectorWorths) # no persistence applied to this
    average_worth = a2.mean()
    a4[:] = (a3 - a3.mean())/(a3.max() - a3.min() + EPSILON)
    a5[:] = (a5 - a5.mean())/(a5.max() - a5.min() + EPSILON)
    a6[:] = (a2 - average_worth)/(a2.max() - a2.min() + EPSILON)
    return average_worth


def compute_sectorized_states(listOfBerries:ndarray, 
                        berry_worth_function:Callable, maxPossibleDist:float,
                        prev_sectorized_state:ndarray=None, 
                        persistence=0.8,angle=45
) -> Tuple[List[ndarray], float, ndarray]:
    """ 
    ### parameters
    - listOfBerries: ndarray
            - observation returned from BerryFieldEnv.step
            - a numpy array with rows as [x,y,berry-size]
            - x,y is the RELATIVE position of berry from agent 
    - berry_worth_function: function
            - a function that takes in the sizes (array) 
            and distances (array) and returns a array
            denoting the worth of each berry. 
    - maxPossibleDist: float
            - the maximum possible distance to a berry
            - used to normalize the distance of berry from agent
    - prev_sectorized_state: ndarray (default None)
            - previously computed sectorized state
            to be used for persistence of vision
            - if None, then persistence has no effect
    - persistence: float (default 0.8)
            - the amount of information from the 
            prev_sectorized_state that is retained 
            - not applied on all the features
            
    ### returns
    - sectorized_state: ndarray
    - avg_worth: float, the average worth so seen"""

    # apply persistence if prev_sectorized_state is given
    if prev_sectorized_state is None:
        prev_sectorized_state = np.zeros((6,360//angle))
    prev_sectorized_state = persistence * prev_sectorized_state
    
    # a1: max-worth of each sector
    # a2: stores avg-worth of each sector
    # a3: a mesure of distance to max worthy in each sector
    # a4: normalized mesure of distance to max worthy in each sector
    # a5: normalized population of berries in each sector
    # a6: normalized average-worth of each sector

    if len(listOfBerries) == 0:
        return prev_sectorized_state, 0, np.array([])
        
    sizes = listOfBerries[:,2]
    dist = np.linalg.norm(listOfBerries[:,:2], axis=1) + EPSILON
    directions = listOfBerries[:,:2]/dist[:,None]
    angles = njitGetTrueAngles(directions)
    worths = berry_worth_function(sizes, dist)
    avg_worth =  njitSectorized(
        angles, worths, dist, prev_sectorized_state, angle, maxPossibleDist
    )

    return prev_sectorized_state, avg_worth, worths


if __name__ == "__main__":
    directions = np.random.randn(100, 2)
    directions = directions/np.linalg.norm(directions, axis=1)[:,None]
    # directions = np.asfarray([[0,1],[1,0],[0,-1],[-1,0]])
    angles = njitGetTrueAngles(directions)

    xy = np.random.randint(-1300, 1300, size=(80,2))
    sizes = np.random.randint(1,5, size=80)*10
    berries = np.column_stack([xy[:,0], xy[:,1], sizes])
    
    from berry_worth_function import berry_worth
    berryWorthFunc = lambda sizes, distances: berry_worth(sizes, distances, REWARD_RATE=1e-4, DRAIN_RATE=1/(2*120*400), HALFDIAGOBS = 0.5*(1920**2 + 1080**2)**0.5)
    state, avgWorth, worths = compute_sectorized_states(berries, None, berryWorthFunc, 0.5*(1920**2 + 1080**2)**0.5)