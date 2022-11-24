# the env rewards seen by the agent may be said to be its perception 
# of the env. And there are complex actions that take multiple steps
# in the env, but appear as a single action/Qvalue to the neural-network 
# model inside the agent. Thus there is a need of an intermediate between
# the bare-bones neural-model and the environment. 
import numpy as np
from typing import Callable
from berry_field.envs import BerryFieldEnv
from .complex_actions.explorative_actions import RandomExplorationAction
from .memory_manager import MemoryManager
from .state_computation import StateComputation
from .reward_perception import RewardPerception
from .complex_actions.utils import skip_steps

class BerryFieldEnvAdapter:
    def __init__(
            self, 
            random_exploration_action: RandomExplorationAction,
            reward_perception: RewardPerception,
            memory_manager: MemoryManager, 
            state_computation: StateComputation,
            on_env_reset: Callable,
            skip_steps = 10
    ) -> None:
        self.default_n_actions = len(BerryFieldEnv.ACTION_SWITCHER)
        self.random_exploration_action = random_exploration_action
        self.memory_manager = memory_manager
        self.state_computation = state_computation
        self.reward_perception = reward_perception
        self.skip_steps = skip_steps
        self.on_env_reset = on_env_reset
        self._init_stats()

    def get_stats(self):
        return {
            "action_stats": {i:x for i,x in enumerate(self.action_stats)}
        }

    def resetAdapter(self):
        self.action_stats[:] = 0

    def create_adapter_for_env(self, berry_env: BerryFieldEnv):
        berry_env.reset = self._make_env_reset_wrapper(berry_env.reset)
        berry_env.step = self._make_env_step_wrapper(berry_env.step)

    def _make_env_step_wrapper(self, berry_env_step: Callable):
        def step(action:int):
            if action < self.default_n_actions:
                sum_reward, skip_trajectory, steps = \
                    skip_steps(action, self.skip_steps, berry_env_step)
            else:
                sum_reward, skip_trajectory, steps = \
                    self.random_exploration_action.start_for(berry_env_step)
            _, info, _, done = skip_trajectory[-1] # obs, info, reward, done
            reward = self.reward_perception.get_perceived_reward(sum_reward)
            self.action_stats[action] += 1
            state = self.state_computation.compute(skip_trajectory, action)
            return state, reward, done, info
            
        return step

    def _make_env_reset_wrapper(self, berry_env_reset: Callable):
        def reset():
            observation = berry_env_reset()
            self.on_env_reset() # also resets state_computation
            return self.state_computation.compute([[observation]], -1)
        return reset

    def _init_stats(self):
        self.action_stats = np.zeros(self.default_n_actions + 1)