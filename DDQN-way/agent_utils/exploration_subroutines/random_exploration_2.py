""" This one artificially moves the agent closer
to berry once visible. Not great!!! Hence never used """

import numpy as np
from berry_field.envs import BerryFieldEnv
from berry_field.envs.utils.misc import getTrueAngles

def random_exploration(berryenv:BerryFieldEnv, discount=1.0, render=False, renderS=10):
    """ the reward may be discounted using discount arg """

    if berryenv.action_space.n > 8:
        print("WARNING: assumed that first 8 actions correspond to",
        "directions - N, NE, E, SE, S, SW, W, NW ")

    nactions = 8
    preventive_actions = np.array([2,6,4,0])
    p_action = np.random.randint(8)
    berry_env_step= berryenv.step 

    def move_towards(direction):
        direction = direction/np.linalg.norm(direction)
        angle = getTrueAngles([direction])[0]
        action = int(angle//45)
        return action

    print('initial p_action:', p_action)
    def subroutine(nsteps=1E10,**kwrgs):
        nonlocal p_action
        reward_ = 0
        discount_ = 1

        # p_action = berryenv.current_action
        act_ = np.zeros(nactions)
        act_[p_action] = 1
        
        steps = 0
        listberries = []
        current_patch = None
        for i in range(int(nsteps)):

            # sample an action
            action = np.random.choice(nactions, p=act_)

            listberries, reward, done, info = berry_env_step(action)

            # update the discounted reward in exploration
            reward_ += reward * discount_
            discount_ *= discount
            steps += 1
            current_patch = info['current-patch-id']

            # modify action: if wall in view, avoid hitting
            mask = np.array(info['scaled_dist_from_edge']) < 0.5; s = sum(mask)
            if s > 0: action = np.dot(mask, preventive_actions)//s

            # modify action: saw berry but currently not in patch
            if current_patch is None and len(listberries) != 0: 
                action = move_towards(listberries[0][:2])

            # update action probs
            act_[p_action]=act_[(p_action+1)%8]=act_[(p_action-1)%8]=0
            act_[action]=0.999
            act_[(action+1)%8]=act_[(action-1)%8]=(1-act_[action])/2
            p_action = action

            if not done and render and steps%renderS==0: berryenv.render()
            if current_patch is not None: break
            if done: break
        
        return steps, listberries, reward_, done, info

    return subroutine