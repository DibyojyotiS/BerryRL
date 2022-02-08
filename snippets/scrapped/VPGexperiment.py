from DRLagents import epsilonGreedyAction, greedyAction, softMaxAction
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from get_env import make_berryField
from DRLagents import VPG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# net to score each berry in the view
# input-shape: ndarray of shape (x,2)
class sharedModel(nn.Module):
    def __init__(self):
        super(sharedModel, self).__init__()
        self.linear1 = nn.Linear(77, 32)
        self.linear2 = nn.Linear(32, 16)

    def forward(self, x:Tensor):
        x = x.view(-1, 77)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

class policyModel(nn.Module):
    def __init__(self, sharedmodule:nn.Module) -> None:
        super(policyModel, self).__init__()
        self.sharedmodule = sharedmodule
        self.policylayer = nn.Linear(16, 9)

    def forward(self, x):
        x = self.sharedmodule(x)
        x = self.policylayer(x)
        x = F.log_softmax(x, dim=-1)
        return x

class valueModel(nn.Module):
    def __init__(self, sharedmodule) -> None:
        super(valueModel, self).__init__()
        self.sharedmodule = sharedmodule
        self.valuelayer = nn.Linear(16,1)

    def forward(self, x):
        x = self.sharedmodule(x)
        x = self.valuelayer(x)   
        return x     
        

# training
sharedmodel = sharedModel().to(device)
policymodel = policyModel(sharedmodel).to(device)
valuemodel  = valueModel(sharedmodel).to(device)

voptim = RMSprop(valuemodel.parameters(), lr=0.01)
poptim = RMSprop(policymodel.parameters(), lr=0.01)
tstrat = softMaxAction(policymodel,outputs_LogProbs=True)

def makestate(obs:np.ndarray, infos):
    obs = obs[-1].flatten()
    linf = infos[-1]
    edged = [10000]*4 if linf is None else linf['dist_from_edge']
    cumlr = [0.5 if linf is None else linf['cummulative_reward']]
    return np.concatenate([obs, edged, cumlr])

# easier env
env = make_berryField(observation_type='buckets', agent_size=100, bucket_angle=10)
agent = VPG(env, policymodel, valuemodel, tstrat, poptim, voptim, makestate, gamma=0.99,
                MaxTrainEpisodes=500, MaxStepsPerEpisode=12000, beta=0.1, value_steps=100,
                trajectory_seg_length=2000, skipSteps=20, printFreq=1, device= device)
trianHist = agent.trainAgent(render=False)
evalHist = agent.evaluate(tstrat, 10, True)