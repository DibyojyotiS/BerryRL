import time
import wandb
from DRLagents import *
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from torch.optim.lr_scheduler import MultiStepLR

from Agent import *
from utils import (getRandomEnv, Env_print_fn, 
        my_print_fn, copy_files, plot_berries_picked_vs_episode)

# set all seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# high level configs
WANDB_CALLBACK = False
LOG_DIR = os.path.join('.temp' , '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6]))
RESUME_DIR= LOG_DIR
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper functions
def random_train_env(log_dir):
    print('random_train_env')
    trainEnv = getRandomEnv(field_size=(20000,20000), 
        patch_size=(2600,2600), num_patches=10, seperation=2400, 
        nberries=80, spawn_radius=100, 
        patch_with_agent_at_center=True,
        penalize_boundary_hit=False, logDir=log_dir)
    return trainEnv

if __name__ == '__main__':

    # some variables used later
    resume_run = os.path.exists(RESUME_DIR)
    callbacks = []

    # init wandb callback if enabled
    if WANDB_CALLBACK:
        wandb.init(project="test-project", entity="foraging-rl")
        callbacks.append(wandb.log)

    # copy all files into log-dir and setup logging
    logger = StdoutLogger(filename=f'{LOG_DIR}/log.txt')
    abs_file_dir = os.path.split(os.path.abspath(__file__))[0]
    copy_files(abs_file_dir, f'{LOG_DIR}/pyfiles-backup')

    # setup eval and train env
    trainEnv = random_train_env(LOG_DIR)
    evalEnv = BerryFieldEnv(analytics_folder=f'{LOG_DIR}/eval')
    
    # make the agent and network and wrap evalEnv's step fn
    agent = Agent(trainEnv, skipStep=10, nstep_transition=[1], device=TORCH_DEVICE)
    evalEnv.step = agent.env_step_wrapper(evalEnv)
    nnet = agent.getNet(); print(nnet)

    # init training prereqs
    optim = Adam(nnet.parameters(), lr=0.005, weight_decay=0.05)
    schdl = MultiStepLR(optim,[50*i for i in range(1,21)],gamma=0.5)
    buffer = PrioritizedExperienceRelpayBuffer(bufferSize=int(6E4), alpha=0.95,
                                        beta=0.1, beta_rate=0.9/2000)
    tstrat = epsilonGreedyAction(epsilon=0.5,finalepsilon=0.2,decaySteps=1000)

    ddqn_trainer = DDQN(trainEnv, nnet, tstrat, optim, buffer, batchSize=512, 
                        gamma=0.9, update_freq=5, MaxTrainEpisodes=2000, lr_scheduler=schdl,
                        optimize_every_kth_action=100, num_gradient_steps=25,
                        make_state=agent.makeState, make_transitions=agent.makeStateTransitions,
                        evalFreq=10, printFreq=1, polyak_average=True, polyak_tau=0.1,
                        log_dir=LOG_DIR, resumeable_snapshot=10, device=TORCH_DEVICE)

    # print some of the settings for easier reference
    print(f'optim = {ddqn_trainer.optimizer}, num_gradient_steps= {ddqn_trainer.num_gradient_steps}')
    print(f"optimizing the online-model after every {ddqn_trainer.optimize_every_kth_action} actions")
    print(f"batch size={ddqn_trainer.batchSize}, gamma={ddqn_trainer.gamma}, alpha={buffer.alpha}")
    print(f"polyak_tau={ddqn_trainer.tau}, update_freq={ddqn_trainer.update_freq_episode}")

    # try to resume training
    if resume_run: ddqn_trainer.attempt_resume(RESUME_DIR)    

    # make print-fuctions to print extra info
    trainPrintFn = my_print_fn(trainEnv, buffer, tstrat, ddqn_trainer, schdl)
    evalPrintFn = lambda : Env_print_fn(evalEnv)

    # start training! :D
    try: trianHist = ddqn_trainer.trainAgent(evalEnv=evalEnv,
            train_printFn=trainPrintFn, eval_printFn=evalPrintFn, 
            training_callbacks=callbacks)
    except KeyboardInterrupt as kb: pass

    # final evaluation with render
    ddqn_trainer.evaluate(evalEnv=evalEnv, render=True)
    evalPrintFn()
    logger.close()

    plot_berries_picked_vs_episode(LOG_DIR)