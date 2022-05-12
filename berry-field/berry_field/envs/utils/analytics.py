# background process to save the analytics

# analytics to be saved (for each and every epoch (resets))
# 1. time spent in peripheri and center of each patch
# 2. time spent exploring
# 3. the trajectory followed (x,y) coordinates
# 4. the sequence of actions taken
# 5. when berries are collected

import os
import numpy as np
from berry_field.envs.berry_field_env import BerryFieldEnv


class BerryFieldAnalytics():
    def __init__(self, berryField:BerryFieldEnv, saveFolder:str, verbose:bool=False) -> None:
        """ saves the following to disk
            1. time spent in peripheri and center of each patch
            2. time spent exploring between patches
            3. the trajectory followed (x,y) coordinates
            4. the sequence of actions taken
            5. when berries are collected
            NOTE: construct only after initializing berryField
        """
        self.berryField = berryField
        self.verbose = verbose
        # create files to log the data
        self.agent_path = open(os.path.join(saveFolder,'agent_path.txt'), 'w', encoding='utf-8') # path in (x,y) coordinates
        self.agent_actions = open(os.path.join(saveFolder,'agent_actions.txt'), 'w', encoding='utf-8') # sequence of actions (ints)
        self.berries_along_patch_paths = open(os.path.join(saveFolder,'berries_along_patch_paths.txt'), 'w', encoding='utf-8') # when along patch a berry is collected
        self.patch_data = open(os.path.join(saveFolder,'patch_data.txt'), 'w', encoding='utf-8') # in which patch the agent is; -1 for no patch

        # data-structures to assimilate results
        # to store the time spent in the periperi of patch and the center
        self.num_patches = len(berryField.patch_tree.boxes)
        self.central_patch_times = {patch_no:0 for patch_no in range(self.num_patches)}
        self.peripheral_patch_times = {patch_no:0 for patch_no in range(self.num_patches)}
        # to store the time spent exploring between patches
        self.time_between_patches = 0

        self.unique_berry_sizes = np.unique(berryField.berry_collision_tree.boxes[:,-1])
        self.collected_berries = {patch_no:{x:0 for x in self.unique_berry_sizes} for patch_no in range(self.num_patches)}
        # self.SD_berries = {x:0 for x in self.unique_berry_sizes}
        # self.SD_time_center = 0
        # self.SD_time_peripheri = 0

        # wrte the initial position
        self.agent_path.write(f'{berryField.position},')

        # to compute the preference of the berries
        self.previous_patch_id = -1 # to detect patch-changes
        self.current_patch_steps = 1
        self.berries_along_patch_path = [] # the distance,size of berry collected
        self.total_preference = {x:0 for x in self.unique_berry_sizes}


    def update(self):
        """ log changes """
        # log the new updates
        self.agent_path.write(f'{self.berryField.position},')
        self.agent_actions.write(f'{self.berryField.current_action},')
        self.patch_data.write(f'{self.berryField.current_patch_id},')

        # if we are in the same patch, we may compute the berry preference by
        # noting the distance of the oollected berries along the trajectory line
        patch_id = self.berryField.current_patch_id
        if self.previous_patch_id != patch_id:

            # the agent leaves the previous patch
            if self.previous_patch_id is not None:
                # log the previous berry-along-path
                self.berries_along_patch_paths.write(f'{self.previous_patch_id}:{self.berries_along_patch_path}\n')

                # compute the relative preference
                berry_weight = {size:0 for size in self.unique_berry_sizes}
                berry_count = {size:0 for size in self.unique_berry_sizes}
                for berry_size, distance in self.berries_along_patch_path: 
                    berry_weight[berry_size] += distance
                    berry_count[berry_size] += 1
                for size in self.unique_berry_sizes:
                    if berry_count[size] == 0: continue
                    self.total_preference[size] += berry_weight[size]/berry_count[size]

            # reset variables
            self.current_patch_steps = 1
            self.previous_patch_id = patch_id
            self.berries_along_patch_path = []

        # update time spent
        if self.berryField.current_patch_box is None: self.time_between_patches += 1
        else:
            px,py,pw,ph = self.berryField.current_patch_box
            x,y = self.berryField.position
            x,y = x-px, y-py # shift origin to patch center
            if abs(x) > 0.7 * pw/2 and abs(y) > 0.7 * ph/2: self.peripheral_patch_times[patch_id] += 1
            else: self.central_patch_times[patch_id] += 1 

        # berry related stats
        if len(self.berryField.recently_picked_berries) > 0:
            for size in self.berryField.recently_picked_berries:
                self.collected_berries[patch_id][size] += 1
                self.berries_along_patch_path.append((size, self.current_patch_steps))

        self.current_patch_steps += 1


    def close(self):
        self.agent_path.close()
        self.agent_actions.close()
        self.berries_along_patch_paths.close()
        self.patch_data.close()

        total_peripheral_patch_time = sum([self.peripheral_patch_times[i] for i in range(self.num_patches)])
        total_central_patch_time = sum([self.central_patch_times[i] for i in range(self.num_patches)])

        # print stats
        if self.verbose:
            print("preference: ", self.total_preference)
            print("total peripheral_patch_time: ", total_peripheral_patch_time)
            print("total total_central_patch_time: ", total_central_patch_time)
            print("peripheral_patch_times", self.peripheral_patch_times)
            print("central_patch_times", self.central_patch_times)

        # save the results
        with open('results.txt', 'w') as f:
            f.writelines([
                f"preference: {self.total_preference}",
                f"total peripheral_patch_time: {total_peripheral_patch_time}",
                f"total total_central_patch_time: {total_central_patch_time}",
                f"peripheral_patch_times: {self.peripheral_patch_times}",
                f"central_patch_times: {self.central_patch_times}"
            ])

        
    def compute_stats(self):
        """ compute the stats from the logged information """