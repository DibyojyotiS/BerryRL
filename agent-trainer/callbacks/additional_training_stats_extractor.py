from typing import Dict, Any
from DRLagents.replaybuffers import PrioritizedExperienceRelpayBuffer
from DRLagents.agents.DQN import epsilonGreedyAction
from torch.optim.lr_scheduler import _LRScheduler
from .pipeline import ThreadSafePrinter
import torch
import numpy as np
import wandb

class AdditionalTrainingStatsExtractor:
    def __init__(
        self, per_buffer:PrioritizedExperienceRelpayBuffer, 
        epsilon_greedy_act:epsilonGreedyAction,
        lr_scheduler: _LRScheduler, batch_size:int, 
        wandb_enabled: bool, thread_safe_printer:ThreadSafePrinter
    ) -> None:
        self.per_memory_buffer = per_buffer
        self.epsilon_greedy_act = epsilon_greedy_act
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.wandb_enabled = wandb_enabled
        self.thread_safe_printer = thread_safe_printer

        self.epsilon=1e-6
        self.episode=0

    def __call__(self, info_dict:Dict[str,Any]) -> Dict[str,Any]:
        info_dict["PER_buffer"] = self._get_buffer_infos()
        info_dict["lr"] = self._get_lr()
        info_dict["epsilon"] = self._get_epsilon()
        self._print_train_stuff(info_dict)
        self.episode += 1
        return info_dict

    def _print_train_stuff(self, info_dict:Dict[str,Any]):
        binfo = info_dict["PER_buffer"]
        self.thread_safe_printer(
            f"train episode: {self.episode}\n"
            + f"\t lr: {info_dict['lr']},\n"
            + f"\t epsilon: {info_dict['epsilon']},\n"
            + f"\t buffer: amount_filled: {binfo['amount_filled']},\n"
            + f"\t buffer: alpha: {binfo['alpha']},\n"
            + f"\t buffer: beta: {binfo['beta']},\n"
            + f"\t buffer: num_positive_rewards: {binfo['num_positive_rewards']},\n"
            + f"\t buffer: batch_ratio_positive_rewards: {binfo['batch_ratio_positive_rewards']},\n"
            + f"\t buffer: batch_action_freqs: {binfo['batch_action_freqs']}"
        )

    def _get_lr(self):
        lrAsFloatOrList = self.lr_scheduler.get_lr()
        if type(lrAsFloatOrList) == list and len(lrAsFloatOrList) > 0:
            return lrAsFloatOrList[0]
        return lrAsFloatOrList # type is float
    
    def _get_epsilon(self):
        return self.epsilon_greedy_act.epsilon

    def _get_buffer_infos(self):
        rewards=self.per_memory_buffer.buffer["reward"][:len(self.per_memory_buffer)].squeeze()
        amount_filled=100*len(self.per_memory_buffer)/self.per_memory_buffer.bufferSize
        num_positive_rewards = torch.sum(rewards>=self.epsilon).item()
        
        batch = self.per_memory_buffer.sample(self.batch_size)[0]
        batch_num_positive_rewards = torch.sum(
            (batch["reward"]>=self.epsilon).squeeze()
        ).item()
        batch_actions, batch_action_counts = torch.unique(
            batch['action'], return_counts=True
        )

        bufferinfo = {
            "amount_filled": amount_filled,
            "num_positive_rewards": num_positive_rewards,
            "batch_ratio_positive_rewards": batch_num_positive_rewards/self.batch_size,
            "batch_action_freqs": {
                int(a): int(f) for a,f 
                in zip(
                    self._tensor_to_list(batch_actions), 
                    self._tensor_to_list(batch_action_counts)
                )
            },
            "alpha": self.per_memory_buffer.alpha,
            "beta": self.per_memory_buffer.beta
        }

        if self.wandb_enabled:
            reward_historam = np.histogram(rewards.cpu().numpy(), bins=32)
            bufferinfo["all_rewards_hist"] = \
                wandb.Histogram(np_histogram=reward_historam)

        return bufferinfo

    @staticmethod
    def _tensor_to_list(tensor:torch.Tensor):
        return tensor.cpu().numpy().tolist()