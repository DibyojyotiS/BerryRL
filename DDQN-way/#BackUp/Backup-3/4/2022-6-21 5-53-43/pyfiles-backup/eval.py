from DRLagents import *
from Agent import *
from utils import picture_episode

# set all seeds
seed=10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

LOG_DIR = None
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    berry_env = BerryFieldEnv()
    agent = Agent(berry_env, mode='eval', debug=True, noise=0.03,
                persistence=0.9, time_memory_factor=0.5, skipStep=10, 
                render=True, device=TORCH_DEVICE)

    nnet = agent.getNet()
    nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-760\\onlinemodel_statedict.pt'))
    nnet = nnet.eval()

    buffer = None; optim = None; tstrat = None
    estrat = greedyAction()

    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer,
                        make_state=agent.makeState, log_dir=LOG_DIR, 
                        device=TORCH_DEVICE)
    
    try:ddqn_trainer.evaluate(estrat)
    except KeyboardInterrupt as ex: pass

    print(berry_env.get_numBerriesPicked(), 
        [x for x,v in berry_env.patch_visited.items() if v > 0])

    picture_episode('.temp',0,title='eval')

    agent.showDebug(nnet,f=1)