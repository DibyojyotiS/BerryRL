# background process to save the analytics

# analytics to be saved (for each and every epoch (resets))
# 1. time spent in peripheri and center of each patch
# 2. time spent exploring
# 3. the trajectory followed (x,y) coordinates
# 4. the sequence of actions taken
# 5. when berries are collected

import os
from berry_field.envs.berry_field_env import BerryFieldEnv
import numpy as np


class BerryFieldAnalytics():
    def __init__(self, berryField:BerryFieldEnv, saveFolder:str) -> None:
        """ saves the following to disk
            1. time spent in peripheri and center of each patch
            2. time spent exploring between patches
            3. the trajectory followed (x,y) coordinates
            4. the sequence of actions taken
            5. when berries are collected
            NOTE: construct only after initializing berryField
        """
        self.berryField = berryField
        # create files to log the data
        self.agent_path = open(os.path.join(saveFolder,'agent_path.txt'), 'w') # path in (x,y) coordinates
        self.agent_actions = open(os.path.join(saveFolder,'agent_actions.txt'), 'w') # sequence of actions (ints)
        self.berries_collected = open(os.path.join(saveFolder,'berries_collected.txt'), 'w') # when along path a berry is collected
        self.patch_data = open(os.path.join(saveFolder,'patch_data.txt'), 'w') # in which patch the agent is; -1 for no patch

        # data-structures to assimilate results
        # to store the time spent in the periperi of patch and the center
        self.central_patch_times = {patch_no:0 for patch_no in range(len(berryField.patch_boxes))}
        self.peripheral_patch_times = {patch_no:0 for patch_no in range(len(berryField.patch_boxes))}
        # to store the time spent exploring between patches
        self.time_between_patches = 0

        self.SD_berries = {x:0 for x in np.unique(berryField.berry_collision_tree.boxes[:,-1])}
        self.SD_time_center = 0
        self.SD_time_peripheri = 0


    def update(self, position, related_action):
        """ log changes """
        self.agent_path.write(f'{position},')
        self.agent_actions.write(f'{related_action}')
        self.berries_collected.write(f'')
