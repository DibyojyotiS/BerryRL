from typing import Tuple
import numpy as np
from torch import argmax, float32, nn, tensor
from berry_field.envs import BerryFieldEnv
from .utils import skip_steps

def random_exploration_v2(berryenv:BerryFieldEnv, model:nn.Module, 
        makeState:'function', hasInternalMemory=True, 
        skipSteps=10, discount=1.0, device=None, render=False, 
        *args,**kwargs):
    """ The exploration will start only when there is no berry 
    in view and a random strategy is used until a berry is seen.
    When a berry is seen, the given model is used to infer the 
    best action and we step using that action until we are inside
    a patch. The subroutine terminates when we enter a patch or 
    when the berryend terminates the episode.

    ### params
    1. berryenv: BerryFieldEnv
            - the step function from this instance is used
    2. makeState: function
            - the makeState function of the agent
            - computed states used for guided exploration
            - also it is intended that the call to it will 
            update the agent's internal memory.
    3. hasInternalMemory: bool (default True)
            - wether the agent has internal memory
            - if false then makeState will not be called during
            the purely random exploration part of subroutine
    3. skipSteps: int (default=10)
            - any action is repeated skipSteps times
    4. discount: float (default 1.0)
            - The reward may be discounted using discount arg. 
    5. device: device (default None)
            - the device where the input to model is 
            transfered to before calling model with inputs
    6. render: bool (default False) renders the env after frameskipping""" 

    pmx = 0.994
    nactions = 8
    berryenv_step= berryenv.step

    preventive_actions = np.array([2,6,4,0])
    p_action = np.random.randint(nactions)
    probs = np.zeros(nactions)
    probs[p_action] = 1.0

    print('p_action:', p_action)

    def update_action_probs(action):
        nonlocal p_action, probs
        probs[p_action]=probs[(p_action+1)%8]=probs[(p_action-1)%8]=0
        probs[(action+1)%8]=probs[(action-1)%8]=(1-pmx)/2
        probs[action]=pmx
        p_action = action

    def rnd_exploration_step():
        nonlocal probs
        action = np.random.choice(nactions, p=probs)
        sum_reward, skip_trajectory, steps = \
            skip_steps(action, skipSteps, berryenv_step)

        # call makeState to update the agent's memory
        if hasInternalMemory: makeState(skip_trajectory, action)

        # if wall in view, avoid hitting
        edge_dist=skip_trajectory[-1][1]['scaled_dist_from_edge']
        mask = np.array(edge_dist) < 0.5; s = sum(mask)
        if s > 0: action = np.dot(mask, preventive_actions)//s

        update_action_probs(action)

        return sum_reward, skip_trajectory, steps

    def guided_exploration_step(skip_trajectory):
        state = makeState(skip_trajectory, p_action)
        qvals = model(tensor([state], dtype=float32, device=device))
        action = argmax(qvals[:,:8], dim=-1, keepdim=True).item()
        # berryenv.render() # uncomment for debugging purposes
        return skip_steps(action, skipSteps, berryenv_step)     

    def subroutine(*args, **kwargs) -> Tuple[int, float, list]:
        """ returns discounted_reward, skip_trajectory, total_steps"""
        steps_ = reward_ = 0; discount_= 1    
        
        # init skip trajectory (subroutine should take atleast one step)
        sum_reward, skip_trajectory, steps = rnd_exploration_step()
        steps_+=steps; reward_ += sum_reward*discount_; discount_*=discount
        listberries, info, _, done = skip_trajectory[-1]

        while not done and (len(listberries) == 0):
            if render: berryenv.render()
            sum_reward, skip_trajectory, steps = rnd_exploration_step()
            steps_+=steps; reward_ += sum_reward*discount_; discount_*=discount
            listberries, info, _, done = skip_trajectory[-1]
            current_patch = info['current-patch-id']

            while not done and (len(listberries) != 0) and current_patch is None:
                if render: berryenv.render()
                sum_reward, skip_trajectory, steps = guided_exploration_step(skip_trajectory)
                steps_+=steps; reward_ += sum_reward*discount_; discount_*=discount
                listberries, info, _, done = skip_trajectory[-1]
                current_patch = info['current-patch-id']

        return reward_, skip_trajectory, steps_

    return subroutine