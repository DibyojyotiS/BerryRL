from berry_field.envs.utils.misc import getTrueAngles
import numpy as np

from unordered_observation import unordered_observation

BUCKET_ANGLE = 45
OBSHAPE = 40 # observation rows

# for buckets observation
half_bucket_angle = BUCKET_ANGLE/2
sectors = [((x-half_bucket_angle)%360, x+half_bucket_angle) for x in range(0,360,BUCKET_ANGLE)] # for observatiof bucket type
NUMBUCKETS = len(sectors)

def bucket_obseration(raw_observation, agentx, agenty, OBSHAPE = 40):
    """ berries averaged into buckets representing directions
        returns np array of shape (OBSHAPE, 2)
        1st column: num of sizes of beries in a bucket
        2nd column: average distance to berries in a bucket """
    observation = np.zeros((NUMBUCKETS, 2))
    obs = unordered_observation(raw_observation, agentx, agenty, OBSHAPE)
    berries = np.argwhere(np.isclose(obs[:,0], 1))[:,0]

    if berries.shape[0]==0: return observation
    obs = obs[berries]
    directions, distances, sizes = obs[:,1:3], obs[:,3], obs[:,4]
    angles = getTrueAngles(directions, [0,1])

    for i, sector in enumerate(sectors):
        if sector[0] < sector[1]:
            args = np.argwhere((angles>=sector[0])&(angles<=sector[1]))
        else:
            args = np.argwhere((angles>=sector[0])|(angles<=sector[1]))
        
        # if no berries in sector
        if args.shape[0] == 0: continue

        args = np.squeeze(args)
        observation[i,0] = np.mean(sizes[args])    
        observation[i,1] = np.mean(distances[args])      

    return observation  