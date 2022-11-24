from copy import deepcopy
from typing import Any, Dict

from berry_field import BerryFieldEnv

from agent import Agent

from .pipeline import DaemonPipe, ThreadSafePrinter
from .env_analytics_processor import BerryFieldAnalyticsProcessor
from .qvalues_stats import QvalueStats
from .logger_saver import LoggerSaver
from .agent_analytics_processor import AgentAnalyticsProcessor


class StatsCollectorAndLoggerCallback:
    """ collects stats from the agent and from the environments """
    def __init__(
        self, 
        agent: Agent,
        berry_env: BerryFieldEnv, 
        save_dir:str,
        tag:str,
        wandb: bool,
        thread_safe_printer: ThreadSafePrinter,
        episodes_per_video: int
    ) -> None:
        self.agent = agent
        self.berry_env = berry_env
        self.save_dir = save_dir
        self.tag = tag
        self.wandb = wandb
        self.thread_safe_printer = thread_safe_printer
        self.episodes_per_video = episodes_per_video
        self.daemon_pipe = self._init_daemon_pipe()

    def close(self):
        self.daemon_pipe.stop()

    def __call__(self, info_dict):
        info_dict["raw_stats"] = self._collect_stats()
        self.daemon_pipe(info_dict)

    def _collect_stats(self):
        stats = {
            "agent": deepcopy(self.agent.get_stats()),
            "env": deepcopy(self.berry_env.get_analysis())
        }
        return stats

    def _init_daemon_pipe(self):
        analytics = BerryFieldAnalyticsProcessor(
            berry_field=self.berry_env,
            save_dir=self.save_dir,
            wandb=self.wandb,
            episodes_per_video=self.episodes_per_video
        )
        qval_stats = QvalueStats()
        agent_analytics = AgentAnalyticsProcessor()
        logger = LoggerSaver(
            save_dir=self.save_dir,
            tag=self.tag,
            wandb=self.wandb,
            thread_safe_printer=self.thread_safe_printer
        )
        return DaemonPipe([
            analytics,
            agent_analytics,
            qval_stats,
            logger
        ])

    