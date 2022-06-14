import numpy as np
from berry_field.envs import BerryFieldEnv
from torch import from_numpy, softmax

def exploration_routine(berryenv:BerryFieldEnv, discount=1.0, temp=0.09, render=False):
    """ the reward may be discounted using discount arg """

    if berryenv.action_space.n > 8:
        print("WARNING: assumed that first 8 actions correspond to",
        "directions - N, NE, E, SE, S, SW, W, NW ")

    nactions = 8
    preventive_actions = np.array([2,6,4,0])
    p_action = np.random.randint(8)
    berry_env_step= berryenv.step 

    print('initial p_action:', p_action)
    def subroutine(**kwrgs):
        nonlocal p_action
        reward_ = 0
        discount_ = 1

        # p_action = berryenv.current_action
        act_ = np.zeros(nactions)
        act_[p_action] = 1
        
        steps = 0
        listberries = []
        current_patch = None
        for i in range(1000):
        # while (len(listberries) == 0) or current_patch is None:

            # sample an action
            action_probs = softmax(from_numpy(act_),dim=-1)
            action = np.random.choice(nactions, p=action_probs)

            listberries, reward, done, info = berry_env_step(action)
            if render and not done: berryenv.render()

            # update the discounted reward in exploration
            reward_ += reward * discount_
            discount_ *= discount
            steps += 1
            current_patch = info['current-patch-id']

            # if wall in view, avoid hitting
            mask = np.array(info['scaled_dist_from_edge']) < 0.5
            s = sum(mask)
            if s > 0: action = np.dot(mask, preventive_actions)//s

            # update action_
            act_[p_action]=act_[(p_action+1)%8]=act_[(p_action-1)%8]=0
            act_[action]=1/temp
            if s <= 0: act_[(action+1)%8]=act_[(action-1)%8]=0.4/temp
            p_action = action

            if done: break
            if not ((len(listberries) == 0) or current_patch is None): break
        
        return steps, listberries, reward_, done, info

    return subroutine



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def x(a, l=1000):
        INVSQRT2 = 0.5**0.5
        pa = np.zeros(8)
        pa[a]=1
        action_switcher = {
            0: (0, 1), # N
            1: (INVSQRT2, INVSQRT2), # NE
            2: (1, 0), # E
            3: (INVSQRT2, -INVSQRT2), # SE
            4: (0, -1), # S
            5: (-INVSQRT2, -INVSQRT2), # SW
            6: (-1, 0), # W
            7: (-INVSQRT2, INVSQRT2), # NW
        }
        x=10000; y=10000
        pth = [(x,y)]
        ah = []
        tmep = 0.09
        for i in range(l):
            probs = softmax(from_numpy(pa),dim=-1)
            ac = np.random.choice(8, p=probs)
            dx,dy = action_switcher[ac]
            x += dx; y+=dy
            pa[a]=pa[(a-1)%8]=pa[(a+1)%8]=\
                pa[(a-2)%8]=pa[(a+2)%8]=0
            pa[ac] = 1/tmep
            pa[(ac+1)%8]= pa[(ac-1)%8] = 0.4/tmep
            a = ac
            pth.append((x,y))
            ah.append(probs.numpy())

        return pth, ah

    p,ah = x(0,48000)
    p = np.array(p)
    # plt.ylim(0,20000)
    # plt.xlim(0,20000)
    plt.plot(p[:,0],p[:,1])
    plt.show()
    plt.plot(ah)

