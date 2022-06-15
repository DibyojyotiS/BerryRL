from DRLagents import *
from get_baby_env import getRandomEnv
from make_net import *
from Agent import *
from print_utils import picture_episode

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# baby env params
FIELD_SIZE = (20000,20000)
PATCH_SIZE = (2600,2600)
N_PATCHES = 10
SEPERATION= 2400
N_BERRIES = 80

LOG_DIR = None
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # berry_env = getBabyEnv(FIELD_SIZE, PATCH_SIZE, N_PATCHES, SEPERATION, 
    #                         N_BERRIES, LOG_DIR, living_cost=True)
    berry_env = BerryFieldEnv(play_till_maxtime=False)
    agent = Agent(berry_env, mode='eval', debug=True, noise=0.01, persistence=0.8)

    nnet = agent.getNet(TORCH_DEVICE)
    nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-420\\onlinemodel_statedict.pt'))
    nnet = nnet.eval()

    buffer = None; optim = None; tstrat = None
    estrat = softMaxAction(temperature=10) #greedyAction()

    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer, skipSteps=10,
                        make_state=agent.makeState, log_dir=LOG_DIR, device=TORCH_DEVICE)
    
    try:ddqn_trainer.evaluate(estrat, render=1)
    except KeyboardInterrupt as ex: pass

    print(berry_env.get_numBerriesPicked(), 
        [x for x,v in berry_env.patch_visited.items() if v > 0])

    picture_episode('.temp',0)

    agent.showDebug(nnet)