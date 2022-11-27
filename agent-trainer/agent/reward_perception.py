from .memory_manager import MemoryManager

class RewardPerception:
    def __init__(
        self, memory_manager: MemoryManager, 
        max_clip:float=2, min_clip:float=-0.04,
        scale:float=400
    ) -> None:
        self.memory_manager = memory_manager
        # self.max_clip = max_clip
        # self.min_clip = min_clip
        self.scale = scale

    def get_perceived_reward(self, actual_reward):
        if actual_reward > 0:
            picked = self.memory_manager.get_num_berries_picked()
            return picked/100 + self.scale*actual_reward
        else:
            return actual_reward