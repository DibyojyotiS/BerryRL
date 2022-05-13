import time
from DRLagents import *
from torch.optim.rmsprop import RMSprop
from get_baby_env import getBabyEnv
from make_net import *
from State_n_Transition_Maker import *

# baby env params
FIELD_SIZE = (4000,4000)
PATCH_SIZE = (1000,1000)
N_PATCHES = 5
N_BERRIES = 10

# LOG_DIR = os.path.join('.temp/eval/' , '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6]))
LOG_DIR = None
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    berry_env = getBabyEnv(FIELD_SIZE, PATCH_SIZE, N_PATCHES, N_BERRIES, LOG_DIR)
    stMaker = State_n_Transition_Maker(berry_env, mode='eval', positive_emphasis=False)

    nnet = make_net(
        inDim = stMaker.get_output_shape()[0],
        outDim = berry_env.action_space.n,
        hDim = [64,64,64,64,64]
    )
    nnet.load_state_dict(torch.load('Memory_and_LowResPath\path-grid-approach\.temp\\2022-5-13 10-56-57\\2022-5-13 10-57-6\onlinemodel_weights_episode_1.pth'))
    nnet.eval()

    buffer = PrioritizedExperienceRelpayBuffer(int(1E5), 0.95, 0.1, 0.01)
    optim = RMSprop(nnet.parameters(), lr=0.0001)
    tstrat = epsilonGreedyAction(nnet, 0.5, 0.01, 50)
    estrat = softMaxAction(nnet,temperature=1)

    # an user-print-function to print extra stats
    def print_fn():
        if buffer.buffer is not None:
            visited_patches = [p for p in berry_env.patch_visited.keys() if berry_env.patch_visited[p] > 0]
            print('-> berries picked:', berry_env.get_numBerriesPicked(),
                'of', berry_env.get_totalBerries(), '| patches-visited:', visited_patches, 
                '| positive-in-buffer:', sum(buffer.buffer['reward'].cpu()>0).item())

    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer, batchSize=128, skipSteps=10,
                        make_state=stMaker.makeState, make_transitions=stMaker.makeTransitions,
                        gamma=0.9, MaxTrainEpisodes=50, user_printFn=print_fn,
                        printFreq=1, update_freq=2, polyak_tau=0.8, polyak_average= True,
                        log_dir=LOG_DIR, save_snapshots=True, device=TORCH_DEVICE)
    ddqn_trainer.evaluate(estrat, render=True)