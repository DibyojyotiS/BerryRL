import numpy as np

def berry_worth(sizes:np.ndarray, distances:np.ndarray, 
            REWARD_RATE, DRAIN_RATE, HALFDIAGOBS, WORTH_OFFSET=0,
            min_berry_size=10, max_berry_size=40):
    """ the reward that can be gained by pursuing a berry of given size and distance
    we note that the distances are scaled to be in range 0 to 1 by dividing by half-diag
    of observation space.
    To be noted that berryEnv.step returns observation as a list
    of berries as [x,y,berry-size]. Where x and y are scaled by 
    dividing by the half-diagonal of the observation space. """
    # for computation of berry worth, can help to change 
    # the agent's preference of different sizes of berries. 
    rr, dr = REWARD_RATE, DRAIN_RATE
    worth = rr * sizes - dr * distances * HALFDIAGOBS
    
    # scale worth to 0 - 1 range
    min_worth = rr * min_berry_size - dr * HALFDIAGOBS
    max_worth = rr * max_berry_size
    worth = (worth - min_worth)/(max_worth - min_worth)

    # if there are berries in memory that are outside 
    # the current field of view then worth as above 
    # will be -ve and should be clipped to 0
    worth = np.clip(worth, a_min=0,a_max=None)

    # incorporate offset
    worth = (worth + WORTH_OFFSET)/(1 + WORTH_OFFSET)

    return worth


if __name__ == "__main__":
    REWARD_RATE, DRAIN_RATE, HALFDIAGOBS = \
        1e-4, 1/(2*120*400), 0.5*(1920**2 + 1080**2)**0.5

    sizes = np.array([10,40,10,40])
    dists = np.array([0.99,1.5,0,0])
    w = berry_worth(sizes, dists,
            REWARD_RATE, DRAIN_RATE, HALFDIAGOBS,
            WORTH_OFFSET=0.05)
    print(w)