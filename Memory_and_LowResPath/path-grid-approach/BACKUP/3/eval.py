from DRLagents import *
from torch.optim.rmsprop import RMSprop
from get_baby_env import getBabyEnv
from make_net import *
from Agent import *

# baby env params
FIELD_SIZE = (20000,20000)
PATCH_SIZE = (2600,2600)
N_PATCHES = 5
N_BERRIES = 40

LOG_DIR = None
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    berry_env = getBabyEnv(FIELD_SIZE, PATCH_SIZE, N_PATCHES, N_BERRIES, LOG_DIR, initial_juice=0.01,
                            end_on_boundary_hit=False, allow_no_action=False, show=False)
    agent = Agent(berry_env, mode='eval', debug=True, noise=0.025, persistence=0.7, time_memory_delta=0.001)

    nnet = agent.getNet(TORCH_DEVICE)

    nnet.load_state_dict(torch.load('..\\trainLogs\\onlinemodel_weights_episode_57.pth'))
    nnet.eval()

    buffer = None; optim = None; tstrat = None
    estrat = greedyAction(nnet)

    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer, batchSize=256, skipSteps=10,
                        make_state=agent.makeState, make_transitions=agent.makeStateTransitions,
                        gamma=0.9, MaxTrainEpisodes=50, user_printFn=None,
                        printFreq=1, log_dir=LOG_DIR, save_snapshots=True, device=TORCH_DEVICE)
    ddqn_trainer.evaluate(estrat, render=True)

    agent.showDebug(nnet)