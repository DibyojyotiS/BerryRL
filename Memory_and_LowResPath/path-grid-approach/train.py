import time
import shutil
from DRLagents import *
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from get_baby_env import getBabyEnv
from make_net import *
from Agent import *

# baby env params
FIELD_SIZE = (5000,5000)
PATCH_SIZE = (1400,1400)
N_PATCHES = 5
N_BERRIES = 10

LOG_DIR = os.path.join('.temp' , '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6]))
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    print(LOG_DIR, '\n', 'FIELD_SIZE:', FIELD_SIZE, 'PATCH_SIZE:', 
            PATCH_SIZE, 'N_PATCHES:', N_PATCHES, 'N_BERRIES:', N_BERRIES)

    # copy all files into log-dir and setup logging
    dest = os.path.join(LOG_DIR, 'pyfiles-backup')
    if not os.path.exists(dest): os.makedirs(dest)
    for file in [f for f in os.listdir('.') if f.endswith('.py')]: shutil.copy2(file, dest)
    logger = StdoutLogger(filename=os.path.join(LOG_DIR, 'log.txt'))

    # setup env and model
    berry_env = getBabyEnv(FIELD_SIZE, PATCH_SIZE, N_PATCHES, N_BERRIES, LOG_DIR, living_cost=True, initial_juice=0.1)
    agent = Agent(berry_env)
    nnet = agent.getNet(TORCH_DEVICE); print(nnet)
    
    buffer = PrioritizedExperienceRelpayBuffer(int(2E4), alpha=0.8, beta=0.1, beta_rate=0.01)
    optim = Adam(nnet.parameters(), lr=0.00001)
    tstrat = epsilonGreedyAction(nnet, 0.5, 0.01, 400)
    estrat = greedyAction(nnet)

    # an user-print-function to print extra stats
    def print_fn():
        if buffer.buffer is not None:
            visited_patches = [p for p in berry_env.patch_visited.keys() if berry_env.patch_visited[p] > 0]
            print('-> berries picked:', berry_env.get_numBerriesPicked(),
                'of', berry_env.get_totalBerries(), '| patches-visited:', visited_patches, 
                '| positive-in-buffer:', sum(buffer.buffer['reward'].cpu()>0).item(),
                f'| amount-filled: {100*len(buffer)/buffer.bufferSize:.2f}%')
            print(f'\t| approx positives in sample {256}: {sum(buffer.sample(256)[0]["reward"].cpu()>0).item()}')

    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer, batchSize=256, skipSteps=10,
                        make_state=agent.makeState, make_transitions=agent.makeStateTransitions,
                        gamma=0.99, MaxTrainEpisodes=400, user_printFn=print_fn,
                        printFreq=1, update_freq=1, polyak_tau=0.5, polyak_average= True,
                        log_dir=LOG_DIR, save_snapshots=True, device=TORCH_DEVICE)
    trianHist = ddqn_trainer.trainAgent(render=False)
    ddqn_trainer.evaluate(estrat, render=True)
    logger.close()