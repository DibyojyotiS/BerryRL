from typing import Callable, List
import wandb

class wandbCallback:
    def __init__(self, pipeline:List[Callable]) -> None:
        self.pipeline = pipeline

    def __call__(self, info_dict:dict) -> None:
        for stage in self.pipeline:
            info_dict = stage(info_dict)
        wandb.log(info_dict)