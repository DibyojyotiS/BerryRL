import shutil
import time

from DRLagents import *
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop

from Agent import *
from get_baby_env import getBabyEnv
from make_net import *
from print_utils import my_print_fn

# baby env params
FIELD_SIZE = (20000,20000)
PATCH_SIZE = (2600,2600)
N_PATCHES = 10
N_BERRIES = 80

LOG_DIR = os.path.join('.temp' , '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6]))
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # copy all files into log-dir and setup logging
    logger = StdoutLogger(filename=os.path.join(LOG_DIR, 'log.txt'))
    dest = os.path.join(LOG_DIR, 'pyfiles-backup')
    if not os.path.exists(dest): os.makedirs(dest)
    for file in [f for f in os.listdir('.') if f.endswith('.py')]: shutil.copy2(file, dest)

    # setup env and model and training params
    berry_env = getBabyEnv(FIELD_SIZE, PATCH_SIZE, N_PATCHES, N_BERRIES, LOG_DIR, living_cost=True)
    agent = Agent(berry_env, state_transition_mode='single')
    nnet = agent.getNet(TORCH_DEVICE); print(nnet)
    optim = Adam(nnet.parameters(), lr=0.00005)
    buffer = PrioritizedExperienceRelpayBuffer(int(5E4), alpha=0.95, beta=0.1, beta_rate=0.9/2000)
    tstrat = epsilonGreedyAction(nnet, 0.5, 0.1, 2000)
    estrat = greedyAction(nnet)
    print_fn = my_print_fn(berry_env, buffer, tstrat, 1024)

    print('lr used = 0.00005, num_gradient_steps= 500')
    print("optimizing the online-model after every episode (skipSteps=10)")
    print("batch size-1024, gamma=0.8, alpha=0.95")
    print("polyak_tau=0.1, update_freq=10")
    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer, batchSize=1024, skipSteps=10,
                        make_state=agent.makeState, make_transitions=agent.makeStateTransitions,
                        gamma=0.8, MaxTrainEpisodes=2000, optimize_every_kth_action=-1, printFreq=1,
                        user_printFn=print_fn, polyak_tau=0.1, polyak_average= True, num_gradient_steps= 500,
                        update_freq=10, log_dir=LOG_DIR, save_snapshots=True, device=TORCH_DEVICE)

    try: # train, save optimizer on keyborad interupt
        trianHist = ddqn_trainer.trainAgent(render=False)
    except KeyboardInterrupt as kb:
        torch.save(optim.state_dict(), 
            os.path.join(LOG_DIR,f'optimizer_state.pth'))

    ddqn_trainer.evaluate(estrat, render=True)
    logger.close()