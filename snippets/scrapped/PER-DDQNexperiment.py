from DRLagents import epsilonGreedyAction, greedyAction
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from get_env import make_berryField
from DRLagents import PrioritizedExperienceRelpayBuffer
from DRLagents import DDQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# net to score each berry in the view
# input-shape: nparray of shape (x,3)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(77, 32)
        self.linear2 = nn.Linear(32, 16)
        self.valfn   = nn.Linear(16, 1)
        self.actadvs = nn.Linear(16, 9)

    def forward(self, x:Tensor):
        x = x.view(-1, 77)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        advs = self.actadvs(x)
        val  = self.valfn(x) 
        qvals= val + (advs - advs.mean())
        return qvals

# training
value_net = Model().to(device)
buffer = PrioritizedExperienceRelpayBuffer(int(1E6), 0.9, 0.2, 0.001)
optim = RMSprop(value_net.parameters(), lr=0.01)
tstrat = epsilonGreedyAction(value_net, 0.5, 0.01, 50)
estrat = greedyAction(value_net)

def makestate(obs:np.ndarray, infos):
    obs = obs[-1].flatten()
    linf = infos[-1]
    edged = [10000]*4 if linf is None else linf['dist_from_edge']
    cumlr = [0.5 if linf is None else linf['cummulative_reward']]
    return np.concatenate([obs, edged, cumlr])


# easier env
env = make_berryField(observation_type='buckets', agent_size=100, bucket_angle=10)
agent = DDQN(env, value_net, tstrat, optim, buffer, 512, gamma=0.99, 
                skipSteps=50, make_state=makestate, printFreq=1, device= device)
trianHist = agent.trainAgent(render=False)
evalHist = agent.evaluate(estrat, 10, True)