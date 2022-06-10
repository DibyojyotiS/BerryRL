from DRLagents import *
from get_baby_env import getRandomEnv
from make_net import *
from Agent import *
from print_utils import picture_episode

# baby env params
FIELD_SIZE = (20000,20000)
PATCH_SIZE = (2600,2600)
N_PATCHES = 10
N_BERRIES = 80

LOG_DIR = None
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # berry_env = getBabyEnv(FIELD_SIZE, PATCH_SIZE, N_PATCHES, N_BERRIES, LOG_DIR, #initial_juice=0.1,
    #                         end_on_boundary_hit=False, allow_no_action=False, show=False)
    berry_env = BerryFieldEnv()
    agent = Agent(berry_env, mode='eval', debug=True, noise=0.03, persistence=0.9, time_memory_delta=0.01)

    nnet = agent.getNet(TORCH_DEVICE)

    nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-1659\\onlinemodel_statedict.pt'))
    # nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-1046\\onlinemodel_statedict.pt'))
    # nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-917\\onlinemodel_statedict.pt'))
    # nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-612\\onlinemodel_statedict.pt'))
    # nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-439\\onlinemodel_statedict.pt'))
    # nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-329\\onlinemodel_statedict.pt'))
    # nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-323\\onlinemodel_statedict.pt'))
    # nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-225\\onlinemodel_statedict.pt'))
    # nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-171\\onlinemodel_statedict.pt')) # noise=0.03, persistence=0.9
    nnet = nnet.eval()

    buffer = None; optim = None; tstrat = None
    estrat = greedyAction()

    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer, skipSteps=10,
                        make_state=agent.makeState, log_dir=LOG_DIR, device=TORCH_DEVICE)
    
    try:ddqn_trainer.evaluate(estrat, render=True)
    except KeyboardInterrupt as ex: pass

    print(berry_env.get_numBerriesPicked(), 
        [x for x,v in berry_env.patch_visited.items() if v > 0])

    picture_episode('.temp',0)

    agent.showDebug(nnet)