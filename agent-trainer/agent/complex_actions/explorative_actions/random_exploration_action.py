from typing import Callable

import numpy as np
from torch import argmax, device, float32, nn, tensor

from ...state_computation import StateComputation

from ..utils import skip_steps


class RandomExplorationAction:

    SELECTED_ACTION_PROB = 0.994
    PREVENTIVE_ACTION_ARR = np.array([2,6,4,0])

    def __init__(
        self, torch_model:nn.Module, state_computer: StateComputation,
        n_skip_steps: int, reward_discount_factor=1.0, 
        max_steps:float=float('inf'), torch_device: device= device('cpu')
    ) -> None:
        """ A complex action that takes multiple steps in the environment 
        but appears as a single action to the agent.

        It is assumed that the BerryField has actions:\\
        0 -> North\\
        1 -> North-East\\
        2 -> East\\
        3 -> South-East\\
        4 -> South\\
        5 -> South-West\\
        6 -> West\\
        7 -> North-West\n

        Parameters
        ----------
        torch_model : nn.Module
            _description_
        state_computer : StateComputation
            _description_
        n_skip_steps : int
            _description_
        reward_discount_factor : float, optional
            _description_, by default 1.0
        torch_device : device, optional
            _description_, by default device('cpu')
        """
        self.model = torch_model
        self.device = torch_device
        self.max_steps = max_steps
        self.skipSteps = n_skip_steps
        self.rewardDiscount = reward_discount_factor
        self.state_computer = state_computer
        self.reset()

    def reset(self):
        self.__init_subroutine()
        self.__init_stats()

    def get_stats(self):
        return {
            "num_calls": self.times_called,
            "num_invokes": self.times_called_and_subroutine_invoked
        }

    def start_for(self, berry_env_step: Callable):
        """ returns discounted_reward, skip_trajectory, total_steps"""
        steps_ = reward_ = 0; discount_= 1

        # init skip trajectory (subroutine should take atleast one step)
        summedReward, skipTrajectory, steps = self.__one_exploration_step(berry_env_step)
        steps_+=steps; reward_ += summedReward*discount_; discount_*=self.rewardDiscount
        observation, info, _, done = skipTrajectory[-1]
        listberries = observation["berries"]

        # update stats
        self.times_called +=1
        self.times_called_and_subroutine_invoked += (len(listberries) == 0)

        while not done and (len(listberries) == 0) and steps_ < self.max_steps:
            summedReward, skipTrajectory, steps = self.__one_exploration_step(berry_env_step)
            steps_+=steps; reward_ += summedReward*discount_; discount_*=self.rewardDiscount
            observation, info, _, done = skipTrajectory[-1]
            listberries = observation["berries"]
            current_patch = info['current_patch_id']

            while not done and (len(listberries) != 0) and current_patch is None and steps_ < self.max_steps:
                summedReward, skipTrajectory, steps = self.__guided_exploration_step(skipTrajectory, berry_env_step)
                steps_+=steps; reward_ += summedReward*discount_; discount_*=self.rewardDiscount
                observation, info, _, done = skipTrajectory[-1]
                listberries = observation["berries"]
                current_patch = info['current_patch_id']
        return reward_, skipTrajectory, steps_

    def __init_subroutine(self):
        self.action = np.random.randint(8) # seed with a random action
        self.probs = np.zeros(8)
        self.probs[self.action] = 1

    def __update_probs(self, selectedAction):
        self.probs[self.action]=self.probs[(self.action+1)%8]=self.probs[(self.action-1)%8]=0
        self.probs[(selectedAction+1)%8]=self.probs[(selectedAction-1)%8]=(1-RandomExplorationAction.SELECTED_ACTION_PROB)/2
        self.probs[selectedAction]=RandomExplorationAction.SELECTED_ACTION_PROB

    def __one_exploration_step(self, berry_env_step):
        selectedAction = np.random.choice(8, p=self.probs)
        sum_reward, skip_trajectory, steps = \
            skip_steps(selectedAction, self.skipSteps, berry_env_step)
            
        # if wall in view, avoid hitting
        edge_dist=skip_trajectory[-1][0]['scaled_dist_from_edge']
        mask = np.array(edge_dist) < 0.5; s = sum(mask)
        if s > 0: selectedAction = np.dot(mask, RandomExplorationAction.PREVENTIVE_ACTION_ARR)//s
        
        # update the action probs
        self.__update_probs(selectedAction)
        self.action = selectedAction

        return sum_reward, skip_trajectory, steps

    def __guided_exploration_step(self, skip_trajectory, berry_env_step):
        state = self.state_computer.compute(skip_trajectory, self.action)
        qvals = self.model(tensor([state], dtype=float32, device=self.device))
        selectedAction = argmax(qvals[:,:8], dim=-1, keepdim=True).item()
        self.__update_probs(selectedAction)
        self.action = selectedAction
        return skip_steps(self.action, self.skipSteps, berry_env_step)

    def __init_stats(self):
        self.times_called = 0
        self.times_called_and_subroutine_invoked = 0