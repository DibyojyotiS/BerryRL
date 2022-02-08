import numpy as np
import get_env
import matplotlib.pyplot as plt

def getTrueAngles(directions, referenceVector=[0,1]):
    curls = np.cross(directions, referenceVector)
    dot = np.dot(directions, referenceVector)
    angles = np.arccos(dot)*180/np.pi
    args0 = np.argwhere(np.bitwise_not((curls > 0)|(curls == 0)&(dot==1)))
    angles[args0] = 360-angles[args0]
    return angles


sectors = [((x-22.5)%360, x+22.5) for x in range(0,360,45)]
lastaction = 1
def heuristicpolicy(obs, distance_discount=0.8):
    """obs: [[isBerry, direction(2 cols), distance, size]]"""
    global lastaction
    berries = np.argwhere(np.isclose(obs[:,0], 1))[:,0]
    if berries.shape[0]==0: return lastaction

    obs = obs[berries]
    directions, distances, sizes = obs[:,1:3], obs[:,3], obs[:,4]
    angles = getTrueAngles(directions, [0,1])
    juices = np.zeros(8)
    for i, sector in enumerate(sectors):
        if sector[0] < sector[1]:
            args = np.argwhere((angles>=sector[0])&(angles<=sector[1]))
        else:
            args = np.argwhere((angles>=sector[0])|(angles<=sector[1]))
        args = np.squeeze(args)
        juicediscount = np.power(distance_discount, distances[args])
        discounted_juice = np.dot(sizes[args], juicediscount)
        juices[i] = discounted_juice

    action = np.argmax(juices)+1
    lastaction = action
    return action


env = get_env.make_berryField()
step = 0
cummulative_rewards = []
done = False
obs, r, done, info = env.step(0)
while not done:
    a = heuristicpolicy(obs)
    obs, r, done, info = env.step(a)
    cummulative_rewards.append(info['cummulative_reward'])
    step+=1
    env.render()
if env.viewer: env.viewer.close()
print(env.cummulative_reward)
plt.plot(cummulative_rewards) 
plt.show()
