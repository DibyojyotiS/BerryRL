import wandb
import json
import numpy as np
import os
from os.path import join, exists
from .pipeline import ThreadSafePrinter


class LoggerSaver:
    def __init__(
        self, save_dir:str, tag:str, wandb:bool, 
        thread_safe_printer:ThreadSafePrinter
    ) -> None:
        self.save_dir = join(save_dir, "json-data")
        self.tag = tag
        self.wandb_enabled = wandb
        self.thread_safe_printer = thread_safe_printer
        self.episode = 0

        if not exists(self.save_dir):
           os.makedirs(self.save_dir) 

    def __call__(self, info:dict) -> None:
        self.print_stuff(info)
        if self.wandb_enabled:
            self.wandb_log(info)
        self.json_dump(info, f"episode-{self.episode}.json")
        self.episode += 1

    def print_stuff(self, info):
        self.thread_safe_printer(
            f"{self.tag}"
            + f" episode: {self.episode},"
            + f" nberies: {info['env']['berries_picked']},"
            + f" npatch: {info['env']['num_visited_patches']},"
            + f" steps: {info['env']['env_steps']},"
            + f" action: {info['agent']['action_stats']}"
        )

    def json_dump(self, info:dict, filename:str):
        save_dir = join(self.save_dir, f"episode-{self.episode}")
        if not exists(save_dir):os.makedirs(save_dir)
        info = self._extract_and_save_np_arrays(save_dir, info)
        def default(obj):
            return "<not serialized>"
        with open(join(save_dir, filename), 'w') as fp:
            json.dump(info, fp, default=default, indent=2)

    def wandb_log(self, info:dict):
        raw_stats = info.pop("raw_stats", None)
        logging_dict = {self.tag: info}
        if info.get("trainEpisode", None) is not None:
            logging_dict["step"] = info["trainEpisode"]
        wandb.log(logging_dict)
        if raw_stats is not None: info["raw_stats"] = raw_stats

    def _extract_and_save_np_arrays(self, save_dir, info:dict, path=""):
        for key in info.keys():
            if type(info[key]) == dict:
                self._extract_and_save_np_arrays(
                    save_dir, info[key], path=f"{path}.{key}"
                )
            else:
                if type(info[key]) == np.ndarray:
                    fpath = join(
                        save_dir, 
                        f"ndarray{path}.{key}.npz"
                    )
                    with open(fpath, 'wb') as fp:
                        np.savez_compressed(
                            fp,
                            **{key: info[key]}
                        )
                    info[key] = fpath
        return info