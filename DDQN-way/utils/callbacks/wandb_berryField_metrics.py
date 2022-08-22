import wandb
from berry_field import BerryFieldEnv

class wandbBerryFieldMetrics:
    def __init__(self,
                 berryField_train: BerryFieldEnv, berryField_eval: BerryFieldEnv
                 ) -> None:
        self.berryField_train = berryField_train
        self.berryField_eval = berryField_eval

    @staticmethod
    def _get_berryFieldEnv_episode_infos_(berry_env: BerryFieldEnv):
        # TODO - Move this function logic in berry_field
        berries_picked = berry_env.get_numBerriesPicked()
        visited_patches = [
            p for p in berry_env.patch_visited.keys()
            if berry_env.patch_visited[p] > 0
        ]
        juice_left = berry_env.total_juice
        return {
            'berries_picked': berries_picked,
            'num_visited_patches': len(visited_patches),
            'juice_left': juice_left
        }

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