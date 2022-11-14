# depreciated!

import numpy as np
from berry_field.envs import BerryFieldEnv

def random_exploration_v1(berryenv:BerryFieldEnv, discount=1.0, 
                    mode=1, render=False, renderS=10, *args,**kwargs):
    """ Completly random exploration. 
    NOT IMPLEMENTED FOR AGENTS THAT USE INTERNAL MEMORY (like Path Memory)
    ### params
    1. berryenv: BerryFieldEnv
            - the step function from this instance is used
    2. discount: float (default 1.0)
            - The reward may be discounted using discount arg.
    3. mode: int (default 1)
            - mode=1: The exploration will only start when 
            there is no berry in view and stop when a berry is
            in view or when the berryend terminates the episode.
            - mode=2: The exploration will start either when
            there is no berry in view Or when the agent is out of
            a patch & continue until the agent is in a patch AND 
            beries are visible or when the episode terminates. 
            - if exploration does not start then only one step
            is taken along the path of exploration and the subroutine
            is terminated. """

    assert mode in [1,2]

    if berryenv.action_space.n > 8:
        print("WARNING: assumed that first 8 actions correspond to",
        "directions - N, NE, E, SE, S, SW, W, NW ")

    nactions = 8
    preventive_actions = np.array([2,6,4,0])
    p_action = np.random.randint(8)
    berry_env_step= berryenv.step 

    def shouldStop(listberries, current_patch):
        if mode==1 and (len(listberries)!=0): return True
        if mode==2 and not ((len(listberries) == 0) or \
            current_patch is None): return True
        return False

    print('initial p_action:', p_action)
    def subroutine(nsteps=1E10,**kwrgs):
        nonlocal p_action
        reward_ = 0
        discount_ = 1

        act_ = np.zeros(nactions)
        act_[p_action] = 1
        
        steps = 0
        for i in range(int(nsteps)):

            # sample an action
            action = np.random.choice(nactions, p=act_)

            listberries, reward, done, info = berry_env_step(action)

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
            if s <= 0: 
                act_[action]=0.999
                act_[(action+1)%8]=act_[(action-1)%8]=(1-act_[action])/2
            else: act_[action]=1.0
            p_action = action

            if not done and render and steps%renderS==0: berryenv.render()
            if shouldStop(listberries, current_patch): break
            if done: break
        
        return steps, listberries, reward_, done, info

    return subroutine