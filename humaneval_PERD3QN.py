
from berry_field.envs.berry_field_mat_input_env import BerryFieldEnv_MatInput
import torch
from make_state import make_state
from new_script_PERD3QN import make_net
from DRLagents import (DDQN, PrioritizedExperienceRelpayBuffer,
                       epsilonGreedyAction, greedyAction)
from torch.optim.rmsprop import RMSprop

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# making the berry env
berry_env = BerryFieldEnv_MatInput(no_action_r_threshold=0.6, verbose=True)

value_net = make_net(3*8, 9, [16,8,8])

# dummy inits (won't be actually used)
optim = RMSprop(value_net.parameters(), lr=0.001)
tstrat = epsilonGreedyAction(value_net, 0.5, 0.01, 50)
estrat = greedyAction(value_net)
buffer = PrioritizedExperienceRelpayBuffer(int(1E6), 0.9, 0.2, 0.001)


# load weights
value_net.load_state_dict(torch.load('.temp_stuffs\savesPERD3QN\\onlinemodel_weights_episode_25.pth'))
value_net.eval()

agent = DDQN(berry_env, value_net, tstrat, optim, buffer, 128, gamma=0.99, 
                skipSteps=20, make_state=make_state, printFreq=1,
                snapshot_dir=None, device=TORCH_DEVICE)
evalHist = agent.evaluate(estrat, 10, True)
