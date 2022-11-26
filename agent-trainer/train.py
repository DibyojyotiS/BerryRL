import os
import time
import matplotlib
import wandb
from berry_field import BerryFieldEnv
from DRLagents import *

from agent import Agent
from callbacks import StatsCollectorAndLoggerCallback, ThreadSafePrinter
from config import CONFIG
from train_utils import copy_files, getRandomEnv
from DRLagents.agents.DQN import epsilonGreedyAction
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.rmsprop import RMSprop

# set all seeds
set_seed(CONFIG["seed"])
matplotlib.use('Agg')

# constants
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIME_STAMP = '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6])
LOG_DIR = os.path.join(CONFIG["LOG_DIR_ROOT"], TIME_STAMP)

def init_logging(TIME_STAMP, LOG_DIR):
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    stdout_logger = StdoutLogger(os.path.join(LOG_DIR, "log.txt"))
    abs_file_dir = os.path.split(os.path.abspath(__file__))[0]
    copy_files(abs_file_dir, f'{LOG_DIR}/pyfiles-backup')
    # init wandb if enabled
    if CONFIG["WANDB"]["enabled"]:
        wandb.init(
            name=TIME_STAMP,
            project=CONFIG["WANDB"]["project"],
            group=CONFIG["WANDB"]["group"],
            entity=CONFIG["WANDB"]["entity"],
            notes=CONFIG["WANDB"]["notes"],
            dir=LOG_DIR,
            config=CONFIG
        )
    return stdout_logger

if __name__ == "__main__":
    stdout_logger = init_logging(TIME_STAMP, LOG_DIR)

    # setup agent and environments as perceived by agents
    trainEnv:BerryFieldEnv = getRandomEnv(**CONFIG["RND_TRAIN_ENV"])
    evalEnv = BerryFieldEnv()
    agent = Agent(
        berry_env_FIELD_SIZE=trainEnv.FIELD_SIZE,
        berry_env_HALFDIAGOBS=trainEnv.HALFDIAGOBS,
        berry_env_REWARD_RATE=trainEnv.REWARD_RATE,
        berry_env_DRAIN_RATE=trainEnv.DRAIN_RATE,
        torch_device=TORCH_DEVICE,
        **CONFIG["AGENT"]
    )
    trainEnv = agent.getPerceivedEnvironment(trainEnv)
    evalEnv = agent.getPerceivedEnvironment(evalEnv)

    thread_safe_printer = ThreadSafePrinter()
    train_collector = StatsCollectorAndLoggerCallback(
        agent=agent,
        berry_env=trainEnv,
        save_dir=os.path.join(LOG_DIR, "train"),
        wandb=CONFIG["WANDB"]["enabled"],
        tag="train",
        thread_safe_printer=thread_safe_printer,
        episodes_per_video=50
    )

    eval_collector = StatsCollectorAndLoggerCallback(
        agent=agent,
        berry_env=evalEnv,
        save_dir=os.path.join(LOG_DIR, "eval"),
        wandb=CONFIG["WANDB"]["enabled"],
        tag="eval",
        thread_safe_printer=thread_safe_printer,
        episodes_per_video=10
    )

    if CONFIG["WANDB"]["enabled"]:
        wandb.watch(agent.nn_model)

    optim = Adam(params=agent.nn_model.parameters(), **CONFIG["ADAM"])
    schdl = MultiStepLR(optimizer=optim, **CONFIG["MULTI_STEP_LR"])
    buffer = PrioritizedExperienceRelpayBuffer(**CONFIG["PER_BUFFER"])
    tstrat = epsilonGreedyAction(**CONFIG["TRAINING_STRAT_EPSILON_GREEDY"])
    ddqn_trainer = DDQN(
        **CONFIG["DDQN"],
        trainingEnv=trainEnv,
        model = agent.nn_model,
        trainExplorationStrategy=tstrat,
        optimizer=optim,
        lr_scheduler=schdl,
        replayBuffer=buffer,
        log_dir=LOG_DIR,
        device=TORCH_DEVICE
    )

    try:
        trianHist = ddqn_trainer.trainAgent(evalEnv=evalEnv,
                                            training_callback=train_collector,
                                            eval_callback=eval_collector)
    except KeyboardInterrupt as kb:
        pass

    ddqn_trainer.evaluate(evalEnv=evalEnv, render=True)
    train_collector.close()
    eval_collector.close()
    stdout_logger.close()