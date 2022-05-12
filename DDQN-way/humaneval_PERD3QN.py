
import torch
from berry_field.envs.berry_field_env import BerryFieldEnv
from DRLagents import (DDQN, PrioritizedExperienceRelpayBuffer,
                       epsilonGreedyAction, greedyAction, softMaxAction)
from torch.optim.rmsprop import RMSprop

from make_net import make_net
from make_state import get_make_state

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# input_size, make_state_fn = get_make_state(avf=0.8,noise_scale=0.01)
input_size, make_state_fn, _ = get_make_state(avf=0.9, ks=0.0001, kd=0.011473, noise_scale=0.1, offset=0.01)


# making the berry env
berry_env = BerryFieldEnv(no_action_r_threshold=0.6, verbose=True)

value_net = make_net(input_size, 9, [16,8,8])

# dummy inits (won't be actually used)
optim = RMSprop(value_net.parameters(), lr=0.001)
tstrat = epsilonGreedyAction(value_net, 0.5, 0.01, 50)
# estrat = greedyAction(value_net)
estrat = softMaxAction(value_net,temperature=0.1)
buffer = PrioritizedExperienceRelpayBuffer(int(1E6), 0.95, 0.2, 0.001)


# load weights
value_net.load_state_dict(torch.load('.temp_stuffs/w.o-negative-rewards/PERD3QN-state-with-action/onlinemodel_weights_episode_41.pth'))
value_net.eval()

agent = DDQN(berry_env, value_net, tstrat, optim, buffer, 512, gamma=0.99, 
                skipSteps=5, make_state=make_state_fn, printFreq=1, update_freq=2,
                polyak_average=True, polyak_tau=0.2,snapshot_dir=None, device=TORCH_DEVICE)
evalHist = agent.evaluate(estrat, 10, True)
