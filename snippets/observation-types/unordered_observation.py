import numpy as np

def unordered_observation(raw_observation, agentx, agenty, OBSHAPE):
    """ all visible berries are collated as colstack[isBerry, direction, distance, size]
        in the order they had been detected
        returns np array of shape (OBSHAPE,5) """

    observation = np.zeros((OBSHAPE, 5))
    if len(raw_observation) == 0: return observation
    
    agent_pos = np.array([agentx, agenty])
    directions = raw_observation[:,:2] - agent_pos
    distances = np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
    directions = directions/distances
    data = np.column_stack([np.ones_like(distances), directions, distances, raw_observation[:,-1]])
    observation[:data.shape[0],:] = data
    return observation