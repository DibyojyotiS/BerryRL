import numpy as np
import copy
from berry_field.envs.field.field import Field


class BerryStats:
    def __init__(self, n_patches, berrysizes) -> None:
        self.total_berries_picked = 0
        self.patch_steps = 0   
        self.berrysizes = berrysizes
        self.patch_path_berries = []
        self.berries_along_patch_paths = []
        self.berry_collected_per_patch = {
            p:{s:0 for s in berrysizes} for p in range(n_patches)
        }

    def record_berries_picked_along_pach_paths(
        self, patch_id, picked_berries_sizes
    ):
        # record the berry picked size and disance with a patch trajectory
        # while the agemt entered and is inside the patch
        if patch_id is not None:
            self.patch_steps += 1
            self.total_berries_picked += len(picked_berries_sizes)
            for size in picked_berries_sizes:
                self.berry_collected_per_patch[patch_id][size] += 1
                self.patch_path_berries.append((self.patch_steps, size))
        else: # if the agent goes out of patch
            if len(self.patch_path_berries) > 0: 
                self.berries_along_patch_paths.append(self.patch_path_berries)
                self.patch_path_berries = [] # new object
            self.patch_steps = 0

    def compute_preferences(self):
        com_preference, count_preference = \
            self.compute_preferences_by_CoM_and_count()
        return {
            "CoM": com_preference,
            "count": count_preference
        }
             
    def compute_preferences_by_CoM_and_count(self):
        """ compute the berry preferences from the center of mass """

        # compute the relative preference
        berry_weight = {size:0 for size in self.berrysizes}
        berry_count = {size:0 for size in self.berrysizes}
        for patch_path_berries in self.berries_along_patch_paths:
            max_dist = patch_path_berries[-1][0]
            for distance, size in patch_path_berries: 
                berry_weight[size] += distance/max_dist
                berry_count[size] += 1

        com_preference = {size:0 for size in self.berrysizes}
        for size in self.berrysizes:
            if berry_count[size] <= 5: continue
            com_preference[size] = 1 - berry_weight[size]/berry_count[size]

        count_preference = {
            size: berry_count[size]/(1 + self.total_berries_picked)
            for size in self.berrysizes
        }

        return com_preference, count_preference

    def get_berries_collected_per_patch(self):
        return self.berry_collected_per_patch

    def get_berries_picked(self):
        return self.total_berries_picked


class ExplorationStats:
    def __init__(self, n_patches) -> None:
        self.inter_patch_time = 0
        self.visited_patches = np.zeros(n_patches, dtype=np.bool)
        self.patch_periphery_times = np.zeros(n_patches)
        self.patch_central_times = np.zeros(n_patches)
    
    def record(self, patch_id, patch_center_rel):
        if patch_id is None: 
            self.inter_patch_time += 1
        elif patch_center_rel > 0.25:
            self.visited_patches[patch_id] = 1
            self.patch_central_times[patch_id] += 1
        else:
            self.visited_patches[patch_id] = 1
            self.patch_periphery_times[patch_id] += 1

    def get_visited_patches(self):
        return np.where(self.visited_patches)[0]

    def get_patch_periphery_times(self):
        return self.patch_periphery_times

    def get_patch_central_times(self):
        return self.patch_central_times

    def get_inter_patch_time(self):
        return self.inter_patch_time


class BerryFieldAnalitics:
    def __init__(self, field:Field, max_steps: int, ignore_nresets=1) -> None:
        self.decm = 10
        self.reset_count = - ignore_nresets # ignore the first reset
        self.field = field
        self.max_steps = max_steps
        self._init_variables()

    def record_step(self, observation_dict, reward, done, info_dict):
        position = observation_dict["position"]
        total_juice = observation_dict["total_juice"]
        picked_berries = observation_dict["recently_picked_berries"]
        patch_id = info_dict["current_patch_id"]
        patch_center_rel = observation_dict["patch_relative_score"]

        self.explorationStats.record(patch_id, patch_center_rel)
        self.berry_stats.record_berries_picked_along_pach_paths(
            patch_id, picked_berries
        )

        self.path[self.update_count] = position
        self.juice[self.update_count] = total_juice
        self.update_count += 1
        
    def reset(self):
        if not self.was_get_analysis_called and self.reset_count >= 0:
            print("WARNING BerryFieldAnaltics.reset: get_analysis not called!")
        self.reset_count += 1
        self._init_variables()

    def get_analysis(self):
        # TODO support saving of the information as mix of json and npz
        # TODO also save the berry-data and patch-data
        self.was_get_analysis_called = True

        information =  {
            # TODO can there be some metrics from berries_collected_per_patch
            "berries_collected_per_patch": 
                self.berry_stats.get_berries_collected_per_patch(),
            "berry_preferences": self.berry_stats.compute_preferences(),
            "berries_picked": self.berry_stats.get_berries_picked(),
            "visited_patches": self.explorationStats.get_visited_patches(),
            "total_patch_central_time": 
                np.sum(self.explorationStats.get_patch_central_times()),
            "total_patch_periphery_time":
                np.sum(self.explorationStats.get_patch_periphery_times()),
            "inter_patch_time": self.explorationStats.get_inter_patch_time(),
            "sampled_path": self.path[:self.update_count][::10],
            "sampled_juice": self.juice[:self.update_count][::10],
            "berry_boxes": self.init_berry_boxes,
            "patch_boxes": self.init_berry_boxes,
            "env_steps": self.update_count
        }

        return information

    def _init_variables(self):
        # arrays to store stuff, using float16 to save space
        required_len = self.max_steps
        self.juice = np.zeros(required_len, dtype=np.float16)
        self.path = np.zeros((required_len, 2), dtype=np.float32)

        # required for plotting the env-field
        self.init_berry_boxes = self.field.berry_collision_tree.boxes.copy()
        self.init_patch_boxes = self.field.patch_tree.boxes.copy()

        # stats
        n_patches = self.field.get_num_patches()
        berrysizes = self.field.get_unique_berry_sizes()
        self.berry_stats = BerryStats(n_patches, berrysizes)
        self.explorationStats = ExplorationStats(n_patches)

        # resetable vars
        self.update_count = 0
        self.was_get_analysis_called = False