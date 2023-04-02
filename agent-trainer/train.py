import os
import sys
import json
import time
import matplotlib
import wandb
from berry_field import BerryFieldEnv
from DRLagents import *

from agent import Agent
from callbacks import DaemonStatsCollectorAndLoggerCallback, ThreadSafePrinter
from callbacks import AdditionalTrainingStatsExtractor
from train_utils import copy_files, getRandomEnv
from DRLagents.agents.DQN import epsilonGreedyAction
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.rmsprop import RMSprop
from config import prepareConfig

# import builtins
# _print = builtins.print
# def myPrint(*args, **kwargs):
#     _print(*args, **kwargs)
# builtins.print = myPrint

def init_logging(log_dir):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    stdout_logger = StdoutLogger(os.path.join(log_dir, "log.txt"))
    abs_file_dir = os.path.split(os.path.abspath(__file__))[0]
    copy_files(abs_file_dir, f'{log_dir}/pyfiles-backup')
    return stdout_logger


def init_wandb_if_enabled(run_config, run_name, log_dir):
    # init wandb if enabled
    if run_config["WANDB"]["enabled"]:
        wandb.init(
            name=run_name,
            project=run_config["WANDB"]["project"],
            group=run_config["WANDB"]["group"],
            entity=run_config["WANDB"]["entity"],
            notes=run_config["WANDB"]["notes"],
            dir=log_dir,
            config=run_config
        )


def get_run_config(): 
    for arg in sys.argv[1:]:
        arg_name, data = arg.split("=", maxsplit=1)
        if arg_name.strip() == "--run-config":
            data = data.replace("<space>", ' ').replace("<dblQuotes>", '"')
            return prepareConfig(json.loads(data.strip()))
    print("Defaulting to base-config")
    from config import BASE_CONFIG
    return prepareConfig(BASE_CONFIG)


if __name__ == "__main__":
    
    run_config = get_run_config()

    # set all seeds
    set_seed(run_config["seed"])
    matplotlib.use('Agg')

    # run-constants
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_stamp = '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6])
    run_name = f"{run_config['run_name_prefix']} {time_stamp}"
    assert not set(run_name).intersection(["\\","/",":","*","?","\"","<",">"]) # loose validation
    log_dir = os.path.join(run_config["LOG_DIR_ROOT"], run_name)

    stdout_logger = init_logging(log_dir)
    init_wandb_if_enabled(run_config, run_name, log_dir)

    # setup agent and environments as perceived by agents
    trainEnv:BerryFieldEnv = getRandomEnv(**run_config["RND_TRAIN_ENV"])
    evalEnv = BerryFieldEnv()
    agent = Agent(
        berry_env_FIELD_SIZE=trainEnv.FIELD_SIZE,
        berry_env_HALFDIAGOBS=trainEnv.HALFDIAGOBS,
        berry_env_REWARD_RATE=trainEnv.REWARD_RATE,
        berry_env_DRAIN_RATE=trainEnv.DRAIN_RATE,
        torch_device=torch_device,
        **run_config["AGENT"]
    )
    trainEnv = agent.getPerceivedEnvironment(trainEnv)
    evalEnv = agent.getPerceivedEnvironment(evalEnv)

    optim = Adam(params=agent.nn_model.parameters(), **run_config["ADAM"])
    schdl = MultiStepLR(optimizer=optim, **run_config["MULTI_STEP_LR"])
    per_buffer = PrioritizedExperienceRelpayBuffer(**run_config["PER_BUFFER"])
    tstrat = epsilonGreedyAction(**run_config["TRAINING_STRAT_EPSILON_GREEDY"])
    ddqn_trainer = DDQN(
        **run_config["DDQN"],
        trainingEnv=trainEnv,
        model = agent.nn_model,
        trainExplorationStrategy=tstrat,
        optimizer=optim,
        lr_scheduler=schdl,
        replayBuffer=per_buffer,
        log_dir=log_dir,
        device=torch_device
    )

    # init logging dependencies
    thread_safe_printer = ThreadSafePrinter()
    daemon_train_callback = DaemonStatsCollectorAndLoggerCallback(
        agent=agent,
        berry_env=trainEnv,
        save_dir=os.path.join(log_dir, "train"),
        wandb=run_config["WANDB"]["enabled"],
        tag="train",
        thread_safe_printer=thread_safe_printer,
        episodes_per_video=50
    )
    train_additional_info = AdditionalTrainingStatsExtractor(
        per_buffer=per_buffer,
        epsilon_greedy_act=tstrat,
        lr_scheduler=schdl,
        batch_size=run_config["DDQN"]["batchSize"],
        wandb_enabled=run_config["WANDB"]["enabled"],
        thread_safe_printer=thread_safe_printer
    )
    partial_daemon_training_callback = lambda info_dict: \
        daemon_train_callback(train_additional_info(info_dict))

    daemon_eval_callback = DaemonStatsCollectorAndLoggerCallback(
        agent=agent,
        berry_env=evalEnv,
        save_dir=os.path.join(log_dir, "eval"),
        wandb=run_config["WANDB"]["enabled"],
        tag="eval",
        thread_safe_printer=thread_safe_printer,
        episodes_per_video=10
    )

    if run_config["WANDB"]["enabled"]:
        wandb.watch(
            agent.nn_model, 
            log=run_config["WANDB"]["watch_log"],
            log_freq=run_config["WANDB"]["watch_log_freq"]
        )

    try:
        trianHist = ddqn_trainer.trainAgent(evalEnv=evalEnv,
                                            training_callback=partial_daemon_training_callback,
                                            eval_callback=daemon_eval_callback)
    except KeyboardInterrupt as kb:
        pass

    ddqn_trainer.evaluate(evalEnv=evalEnv, render=True)
    daemon_train_callback.close()
    daemon_eval_callback.close()
    stdout_logger.close()