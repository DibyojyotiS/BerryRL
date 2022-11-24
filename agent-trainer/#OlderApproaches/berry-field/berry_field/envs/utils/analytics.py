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
        self.saveFolder = saveFolder
        self.verbose = verbose
        self.closed = False

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

        self.total_berries_collected = 0
        self.unique_berry_sizes = np.unique(berryField.berry_collision_tree.boxes[:,-1]).astype(int)
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
        # self.total_preference = {x:0 for x in self.unique_berry_sizes}


    def update(self, done):
        """ log changes """
        # log the new updates
        self.agent_path.write(f'{self.berryField.position},')
        self.agent_actions.write(f'{self.berryField.current_action},')
        self.patch_data.write(f'{self.berryField.current_patch_id},')

        # if we are in the same patch, we may compute the berry preference by
        # noting the distance of the oollected berries along the trajectory line
        patch_id = self.berryField.current_patch_id
        if (self.previous_patch_id != patch_id) or done:
            # the agent leaves the previous patch
            if self.previous_patch_id is not None:
                # log the previous berry-along-path
                self.berries_along_patch_paths.write(f'{self.previous_patch_id}:{self.berries_along_patch_path}\n')
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

        # berry related stats -- assumes that berries can be collected only in a patch
        if len(self.berryField.recently_picked_berries) > 0:
            for size in self.berryField.recently_picked_berries:
                self.total_berries_collected += 1
                self.collected_berries[patch_id][size] += 1
                self.berries_along_patch_path.append((size, self.current_patch_steps))

        self.current_patch_steps += 1

        # close analysis if done
        if done: self.close()


    def close(self):
        if self.closed: return
        
        self.agent_path.close()
        self.agent_actions.close()
        self.berries_along_patch_paths.close()
        self.patch_data.close()

        total_peripheral_patch_time = sum([self.peripheral_patch_times[i] for i in range(self.num_patches)])
        total_central_patch_time = sum([self.central_patch_times[i] for i in range(self.num_patches)])
        berrycollstr = '\n\t'.join([f'{x}:{z}' for x,z in self.collected_berries.items()])

        stats = self._compute_stats()

        # print stats
        if self.verbose:
            print(f"berry-collection:\n\t{berrycollstr}")
            print("total peripheral_patch_time: ", total_peripheral_patch_time)
            print("total total_central_patch_time: ", total_central_patch_time)
            print("peripheral_patch_times", self.peripheral_patch_times)
            print("central_patch_times", self.central_patch_times)
            print("total berries collected:", self.total_berries_collected)
            print("preferences: ",str(stats['preferences']))

        # save the results
        with open(os.path.join(self.saveFolder,'results.txt'), 'w') as f:
            f.writelines([
                f"berry-collection: \n\t{berrycollstr}\n"
                f"total peripheral_patch_time: {total_peripheral_patch_time}\n",
                f"total total_central_patch_time: {total_central_patch_time}\n",
                f"peripheral_patch_times: {self.peripheral_patch_times}\n",
                f"central_patch_times: {self.central_patch_times}\n",
                f"total berries collected: {self.total_berries_collected}\n",
                f"preferences: ", str(stats['preferences'])
            ])
        
        self.closed = True        

    
    def _compute_preferences(self):
        """ compute the berry preferences from the logged information """
        # open berries_along_patch_paths.txt
        path = os.path.join(self.saveFolder,'berries_along_patch_paths.txt')
        with open(path,'r') as f:
            lines = f.readlines()
        
        # compute the relative preference
        preferences = {size:0 for size in self.unique_berry_sizes}
        berry_weight = {size:0 for size in self.unique_berry_sizes}
        berry_count = {size:0 for size in self.unique_berry_sizes}
        for line in lines:
            patch_, data = line.split(':')
            if patch_ == '-1': continue
            berries_along_patch_path = eval(data)
            for size, distance in berries_along_patch_path: 
                berry_weight[size] += distance
                berry_count[size] += 1
            for size in self.unique_berry_sizes:
                if berry_count[size] <= 5: continue
                preferences[size] += berry_weight[size]/berry_count[size]    

        return preferences       

    
    def _compute_stats(self):
        stats = {'preferences': self._compute_preferences()}
        return stats