import time
import shutil
from DRLagents import *
from torch.optim.rmsprop import RMSprop
from get_baby_env import getBabyEnv
from make_net import *
from Agent import *

# baby env params
FIELD_SIZE = (4000,4000)
PATCH_SIZE = (1000,1000)
N_PATCHES = 5
N_BERRIES = 20

LOG_DIR = os.path.join('.temp' , '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6]))
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # copy all files into log-dir
    dest = os.path.join(LOG_DIR, 'pyfiles-backup')
    if not os.path.exists(dest): os.makedirs(dest)
    for file in [f for f in os.listdir('.') if f.endswith('.py')]: shutil.copy2(file, dest)

    # setup logging
    logger = StdoutLogger(filename=os.path.join(LOG_DIR, 'log.txt'))

    berry_env = getBabyEnv(FIELD_SIZE, PATCH_SIZE, N_PATCHES, N_BERRIES, LOG_DIR)
    agent = Agent(berry_env)

    nnet = agent.getNet(TORCH_DEVICE); print(nnet)

    buffer = PrioritizedExperienceRelpayBuffer(int(2E4), alpha=0.8, beta=0.1, beta_rate=0.01)
    optim = RMSprop(nnet.parameters(), lr=0.0001)
    tstrat = epsilonGreedyAction(nnet, 0.5, 0.01, 50)
    estrat = greedyAction(nnet)

    # an user-print-function to print extra stats
    def print_fn():
        if buffer.buffer is not None:
            visited_patches = [p for p in berry_env.patch_visited.keys() if berry_env.patch_visited[p] > 0]
            print('-> berries picked:', berry_env.get_numBerriesPicked(),
                'of', berry_env.get_totalBerries(), '| patches-visited:', visited_patches, 
                '| positive-in-buffer:', sum(buffer.buffer['reward'].cpu()>0).item(),
                f'| amount-filled: {100*len(buffer)/buffer.bufferSize:.2f}')

    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer, batchSize=256, skipSteps=15,
                        make_state=agent.makeState, make_transitions=agent.makeStateTransitions,
                        gamma=0.9, MaxTrainEpisodes=500, user_printFn=print_fn,
                        printFreq=1, update_freq=2, polyak_tau=0.8, polyak_average= True,
                        log_dir=LOG_DIR, save_snapshots=True, device=TORCH_DEVICE)
    trianHist = ddqn_trainer.trainAgent(render=False)
    ddqn_trainer.evaluate(estrat, render=True)
    logger.close()