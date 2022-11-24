from DRLagents import *
from DRLagents.agents.DDQN import greedyAction
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
    nnet.load_state_dict(torch.load('.temp\\tuning-berry-picked-bool-feature-2.1.2\\2022-10-31 16-56-18\\trainLogs\models\episode-244\\onlinemodel_statedict.pt'))
    nnet = nnet.eval()

    buffer = None; optim = None; tstrat = None
    estrat = greedyAction()

    ddqn_trainer = DDQN(berry_env, nnet, tstrat, optim, buffer,
                        make_state=agent.makeState, log_dir=LOG_DIR, 
                        device=TORCH_DEVICE)
    
    try:ddqn_trainer.evaluate(estrat, render=True)
    except KeyboardInterrupt as ex: pass

    print(berry_env.get_numBerriesPicked(), 
        [x for x,v in berry_env.patch_visited.items() if v > 0])

    picture_episode('.temp',0,title='eval')

    agent.showDebug(nnet,f=1)