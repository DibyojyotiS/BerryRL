import time

from DRLagents import *
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop

from Agent import *
from utils import (getRandomEnv, Env_print_fn, 
            my_print_fn, copy_files)

# set all seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

LOG_DIR = os.path.join('.temp' , '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6]))
RESUME_DIR= LOG_DIR
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_train_env(log_dir):
    print('random_train_env')
    trainEnv = getRandomEnv(field_size=(20000,20000), 
        patch_size=(2600,2600), num_patches=10, seperation=2400, 
        nberries=80, logDir=log_dir, spawn_radius=100, 
        patch_with_agent_at_center=True,
        penalize_boundary_hit=False)
    return trainEnv

def original_train_env(log_dir):
    print('original_train_env')
    trainEnv = BerryFieldEnv(analytics_folder=log_dir, 
                            penalize_boundary_hit=False)
    return trainEnv

if __name__ == '__main__':

    resume_run = os.path.exists(RESUME_DIR)

    # copy all files into log-dir and setup logging
    logger = StdoutLogger(filename=f'{LOG_DIR}/log.txt')
    copy_files('.', f'{LOG_DIR}/pyfiles-backup')

    # setup eval and train env
    trainEnv = random_train_env(LOG_DIR)
    evalEnv = BerryFieldEnv(analytics_folder=f'{LOG_DIR}/eval')
    
    # make the agent and network and wrap evalEnv's step fn
    agent = Agent(trainEnv, skipStep=10, nstep_transition=[1], device=TORCH_DEVICE)
    evalEnv.step = agent.env_step_wrapper(evalEnv)
    nnet = agent.getNet(); print(nnet)

    # training stuffs
    optim = Adam(nnet.parameters(), lr=0.0005)
    buffer = PrioritizedExperienceRelpayBuffer(int(6E4), alpha=0.95,
                                        beta=0.1, beta_rate=0.9/2000)
    tstrat = epsilonGreedyAction(0.5)

    ddqn_trainer = DDQN(trainEnv, nnet, tstrat, optim, buffer, batchSize=512, 
                        gamma=0.9, update_freq=5, MaxTrainEpisodes=2000,
                        optimize_every_kth_action=100, num_gradient_steps=50,
                        make_state=agent.makeState, make_transitions=agent.makeStateTransitions,
                        evalFreq=10, printFreq=1, polyak_average=True, polyak_tau=0.1,
                        log_dir=LOG_DIR, resumeable_snapshot=10, device=TORCH_DEVICE)

    print(f'optim = {ddqn_trainer.optimizer}, num_gradient_steps= {ddqn_trainer.num_gradient_steps}')
    print(f"optimizing the online-model after every {ddqn_trainer.optimize_every_kth_action} actions")
    print(f"batch size={ddqn_trainer.batchSize}, gamma={ddqn_trainer.gamma}, alpha={buffer.alpha}")
    print(f"polyak_tau={ddqn_trainer.tau}, update_freq={ddqn_trainer.update_freq_episode}")
    
    # try to resume training
    if resume_run: ddqn_trainer.attempt_resume(RESUME_DIR)    

    # make print-fuctions to print extra info
    trainPrintFn = my_print_fn(trainEnv, buffer, tstrat, ddqn_trainer)
    evalPrintFn = lambda : Env_print_fn(evalEnv)

    # start training! :D
    try: trianHist = ddqn_trainer.trainAgent(evalEnv=evalEnv,
            train_printFn=trainPrintFn, eval_printFn=evalPrintFn)
    except KeyboardInterrupt as kb: pass

    # final evaluation with render
    ddqn_trainer.evaluate(evalEnv=evalEnv, render=True)
    evalPrintFn()
    logger.close()