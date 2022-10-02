from typing import Callable, List
import wandb

class CallbackPipeline:
    def __init__(self, pipeline:List[Callable]=[]) -> None:
        self.pipeline = pipeline

    def append_stage(self, stage_callable):
        self.pipeline.append(stage_callable)

    def __call__(self, info_dict:dict) -> None:
        for stage in self.pipeline:
            info_dict = stage(info_dict)