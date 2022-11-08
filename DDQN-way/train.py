import time

import wandb
from DRLagents import *
from DRLagents.agents.DQN import epsilonGreedyAction
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.rmsprop import RMSprop

from Agent import *
from config import CONFIG
from utils import (Env_print_fn, copy_files, getRandomEnv, my_print_fn,
                   plot_berries_picked_vs_episode, max_qvalues_stats_callback,
                   wandbEpisodeVideoMaker, CallbackPipeline, 
                   BerryFieldMetricsCallback, InfoPrinterCallback)
from utils.callbacks.callback_pipeline import BackgroundThread

# set all seeds
set_seed(CONFIG["seed"])

# constants
RESUME_DIR = CONFIG["RESUME_DIR"]
ENABLE_WANDB = CONFIG["WANDB"]["ENABLE_WANDB"]

TIME_STAMP = '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6])
LOG_DIR = os.path.join(CONFIG["LOG_DIR_PARENT"], TIME_STAMP)
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_logging():
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    # init wandb if enabled
    if ENABLE_WANDB:
        wandb.init(
            name=TIME_STAMP,
            project=CONFIG["WANDB"]["project"],
            group=CONFIG["WANDB"]["group"],
            entity=CONFIG["WANDB"]["entity"],
            notes=CONFIG["WANDB"]["notes"],
            dir=LOG_DIR,
            config=CONFIG
        )

    # copy all files into log-dir and setup logging
    logger = StdoutLogger(filename=f'{LOG_DIR}/log.txt')
    abs_file_dir = os.path.split(os.path.abspath(__file__))[0]
    copy_files(abs_file_dir, f'{LOG_DIR}/pyfiles-backup')
    return logger

def getDaemonCallbackPipe(trainEnv, evalEnv, nnet):
    # setup wandb callbacks and wandb if enabled
    callback_pipe = CallbackPipeline()
    callback_pipe.append_stage(max_qvalues_stats_callback)
    if ENABLE_WANDB:
        # wandb.watch(nnet, log_freq=CONFIG["WANDB"]["watch_log_freq"])
        wandb_video_mod = wandbEpisodeVideoMaker(
            log_dir=LOG_DIR, save_dir=f'{LOG_DIR}/videos', 
            train_log_freq=10, eval_log_freq=10,
            figsize=(10,10)
        )
        env_metric_mod = BerryFieldMetricsCallback(trainEnv, evalEnv)
        callback_pipe.append_stage(env_metric_mod)
        callback_pipe.append_stage(wandb_video_mod)
        callback_pipe.append_stage(wandb.log)
    daemonCallbackPipe = BackgroundThread(callback_pipe)
    daemonCallbackPipe.startBackGroundProcess()
    training_callbacks = [daemonCallbackPipe]
    return training_callbacks


if __name__ == '__main__':

    logger = init_logging()

    # setup eval and train env
    trainEnv = getRandomEnv(**CONFIG["RND_TRAIN_ENV"], logDir=LOG_DIR)
    evalEnv = BerryFieldEnv(analytics_folder=f'{LOG_DIR}/eval')

    # make the agent and network and wrap evalEnv's step fn
    agent = Agent(**CONFIG["AGENT"], berryField=trainEnv, device=TORCH_DEVICE)
    evalEnv.step = agent.get_wrapped_env_step(evalEnv)
    nnet = agent.getNet()
    print(nnet)

    # init training prereqs
    optim = Adam(params=nnet.parameters(), **CONFIG["ADAM"])
    schdl = MultiStepLR(optimizer=optim, **CONFIG["MULTI_STEP_LR"])
    buffer = PrioritizedExperienceRelpayBuffer(**CONFIG["PER_BUFFER"])
    tstrat = epsilonGreedyAction(**CONFIG["TRAINING_STRAT_EPSILON_GREEDY"])

    ddqn_trainer = DDQN(
        **CONFIG["DDQN"],
        trainingEnv=trainEnv,
        model=nnet,
        trainExplorationStrategy=tstrat,
        optimizer=optim,
        replayBuffer=buffer,
        lr_scheduler=schdl,
        make_state=agent.makeState,
        make_transitions=agent.makeStateTransitions,
        log_dir=LOG_DIR,
        device=TORCH_DEVICE
    )

    # print some of the settings for easier reference
    print(
        f'optim = {ddqn_trainer.optimizer}, num_gradient_steps= {ddqn_trainer.num_gradient_steps}\n'
        + f"optimizing the online-model after every {ddqn_trainer.optimize_every_kth_action} actions\n"
        + f"batch size={ddqn_trainer.batchSize}, gamma={ddqn_trainer.gamma}, alpha={buffer.alpha}\n"
        +f"polyak_tau={ddqn_trainer.tau}, update_freq={ddqn_trainer.update_freq_episode}"
    )

    # try to resume training
    if RESUME_DIR is not None and os.path.exists(RESUME_DIR):
        ddqn_trainer.attempt_resume(RESUME_DIR)

    # make print-fuctions to print extra info
    trainPrintFn = my_print_fn(trainEnv, buffer, tstrat, ddqn_trainer, schdl)
    def evalPrintFn(): return Env_print_fn(evalEnv)

    daemonCallbackPipe = getDaemonCallbackPipe(trainEnv, evalEnv, nnet)
    training_callbacks = [daemonCallbackPipe]

    # start training! :D
    try:
        trianHist = ddqn_trainer.trainAgent(evalEnv=evalEnv,
                                            train_printFn=trainPrintFn, eval_printFn=evalPrintFn,
                                            training_callbacks=training_callbacks)
    except KeyboardInterrupt as kb:
        pass

    
    del daemonCallbackPipe

    # final evaluation with render
    ddqn_trainer.evaluate(evalEnv=evalEnv, render=True)
    evalPrintFn()
    logger.close()

    plot_berries_picked_vs_episode(LOG_DIR)