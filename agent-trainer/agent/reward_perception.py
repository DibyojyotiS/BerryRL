from typing import Dict, Any
from .memory_manager import MemoryManager
from .intrinsic_rewards import PatchDiscoveryReward

class RewardPerception:
    def __init__(
        self, 
        memory_manager: MemoryManager, 
        max_clip:float=2, min_clip:float=-0.04,
        scale:float=400,
        patch_dicovery_reward_config = dict(
            enabled = True,
            reward_value=1.0
        )
    ) -> None:
        self.memory_manager = memory_manager
        self.max_clip = max_clip
        self.min_clip = min_clip
        self.scale = scale
        self._init_rewards(patch_dicovery_reward_config)
        
    def reset(self):
        self._init_vars()
        self.patch_discovery_reward.reset()

    def get_perceived_reward(self, actual_reward:float, info:Dict[str,Any]):
        # (-k*d + j) + J - K*d = 0
        # the agent will find it easier to reduce K*d by using the exploration
        # to avoid this either remove the min-clip or set a low scale
        scaled_clipped_reward = \
            min(self.max_clip, max(self.min_clip, self.scale*actual_reward))
        
        picked = self.memory_manager.get_num_berries_picked()
        if picked > self.last_num_berries_picked:
            self.last_num_berries_picked = picked
            return 1 + picked/100 + scaled_clipped_reward

        if self.patch_discovery_reward_enabled:
            scaled_clipped_reward += self.patch_discovery_reward.updateAndGetReward(info)

        return scaled_clipped_reward

    def _init_vars(self):
        self.last_num_berries_picked = 0

    def _init_rewards(self, patch_dicovery_reward_config:Dict[str,Any]):
        self.patch_discovery_reward_enabled = \
            patch_dicovery_reward_config["enabled"]
        if self.patch_discovery_reward_enabled:
            patch_dicovery_reward_config.pop("enabled")
            self.patch_discovery_reward = \
                PatchDiscoveryReward(**patch_dicovery_reward_config)