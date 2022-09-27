from DRLagents import *
from Agent import *
from utils import picture_episode
from config import CONFIG

# set all seeds
set_seed(CONFIG["seed"])

LOG_DIR = None
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    berry_env = BerryFieldEnv()
    agent = Agent(**CONFIG["AGENT"], berryField=berry_env, device=TORCH_DEVICE)

    nnet = agent.getNet()
    nnet.load_state_dict(torch.load('..\\trainLogs\models\episode-240\\onlinemodel_statedict.pt'))
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