from berry_field.envs.utils.misc import argsort_clockwise
import numpy as np

def ordered_observation(raw_obervation, agentx, agenty, OBSHAPE=40):
    """ unoredered_observation sorted clockwise """
    observation = np.zeros((OBSHAPE, 5))
    if len(raw_obervation) == 0: return observation

    agent_pos = np.array([agentx, agenty])
    directions = raw_obervation[:,:2] - agent_pos
    distances = np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
    directions = directions/distances
    data = np.column_stack([np.ones_like(distances), directions, distances, raw_obervation[:,-1]])
    args = argsort_clockwise(directions)
    observation[:data.shape[0],:] = data[args]
    return observation