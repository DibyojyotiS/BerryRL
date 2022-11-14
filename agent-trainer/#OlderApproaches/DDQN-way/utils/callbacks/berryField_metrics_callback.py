import wandb
from berry_field import BerryFieldEnv

class BerryFieldMetricsCallback:
    def __init__(self,
                 berryField_train: BerryFieldEnv, berryField_eval: BerryFieldEnv,
                 verbose=False
                 ) -> None:
        """ A callable that will print the BerryFieldEnv stats and modifies
        the dict passed in the calling with the stats from the envs. """
        self.berryField_train = berryField_train
        self.berryField_eval = berryField_eval
        self.verbose = verbose

    def _get_berryFieldEnv_episode_infos_(self, berry_env: BerryFieldEnv):
        # TODO - Move this function logic in berry_field
        berries_picked = berry_env.get_numBerriesPicked()
        visited_patches = [
            p for p in berry_env.patch_visited.keys()
            if berry_env.patch_visited[p] > 0
        ]
        juice_left = berry_env.total_juice
        totalBerries = berry_env.get_totalBerries()
        self._print_env_stats(berries_picked, totalBerries, visited_patches, juice_left)
        return {
            'berries_picked': berries_picked,
            'num_visited_patches': len(visited_patches),
            'juice_left': juice_left
        }

    def _print_env_stats(self, berries_picked, totalBerries, visited_patches, juice_left):
        if not self.verbose: return
        print('-> berries picked:', berries_picked, 'of', totalBerries, 
                '| patches-visited:', visited_patches, f'| juice left:{juice_left:.2f}')

    def __call__(self, info_dict: dict):
        # insert the berries picked data in info-dict
        if "train" in info_dict:
            train_infos = self._get_berryFieldEnv_episode_infos_(self.berryField_train)
            for key, val in train_infos.items():
                info_dict["train"][key] = val

        if "eval" in info_dict:
            eval_infos = self._get_berryFieldEnv_episode_infos_(self.berryField_eval)
            for key,val in eval_infos.items():
                info_dict["eval"][0][key] = val

        return info_dict