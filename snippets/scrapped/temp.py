import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from DRLagents import *
from torch import nn, optim

from get_env import make_berryField

# make a gym environment
env = make_berryField(observation_type="buckets", bucket_angle=5)

# custom make-state fn
def make_state(list_of_obs, list_of_infos:'list[dict]'):
    obs = np.concatenate([list_of_obs[0], list_of_obs[-1]]).flatten() # 1st and last instants
    if list_of_infos[0] is not None:
        cumrewards = [list_of_infos[0]['cummulative_reward'],
                    list_of_infos[-1]['cummulative_reward']]
        relcords = [*list_of_infos[0]['relative_coordinates'], 
                    *list_of_infos[0]['relative_coordinates']]
        distedges = [*list_of_infos[0]['dist_from_edge'],
                    *list_of_infos[-1]['dist_from_edge']]
    else:
        cumrewards = [0.5, 0.5]
        relcords = [0,0,0,0]
        distedges = [960, 960, 540, 540, 960, 960, 540, 540]
    state =  np.concatenate([obs, cumrewards, distedges, relcords])
    return state

# pick a suitable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the value network, i know duelling net is not required here but... :p
class duellingDNN(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation = F.relu):
        super(duellingDNN, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDim[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
        self.valuelayer = nn.Linear(hDim[-1], 1)
        self.actionadv = nn.Linear(hDim[-1], outDim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            x = self.activation(layer(x))
        advantages = self.actionadv(x)
        values = self.valuelayer(x)
        qvals = values + (advantages - advantages.mean())
        return qvals


# create the policy network
class net(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation = F.relu):
        super(net, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDim[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
        self.outputlayer = nn.Linear(hDim[-1], outDim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            t = self.activation(layer(t))
        t = self.outputlayer(t)
        t = F.log_softmax(t, -1)
        return t


# # init necessities
# value_model = duellingDNN(inDim=302, outDim=1, hDim=[32,16], activation=F.relu).to(device)
# policy_model = net(inDim=302, outDim=9, hDim=[32,16], activation=F.relu).to(device)
# policyOptimizer = optim.Adam(policy_model.parameters(), lr=0.01)
# valueOptimizer = optim.Adam(value_model.parameters(), lr=0.01)
# # trainExplortionStrategy = softMaxAction(policy_model, outputs_LogProbs=True)
# trainExplortionStrategy = epsilonGreedyAction(policy_model, 0.5, 0.01, 2, outputs_LogProbs=True)
# evalExplortionStrategy = greedyAction(policy_model)

# VPG with baseline failed
# agent = VPG(env, policy_model, value_model, trainExplortionStrategy, policyOptimizer, 
#                 valueOptimizer, make_state=make_state, gamma=0.99, skipSteps=50, MaxTrainEpisodes=50, printFreq=5, device=device)




# D3QN
duellingQnetwork = duellingDNN(inDim=302, outDim=9, hDim=[32,16], activation=F.relu).to(device)
optimizer = optim.Adam(duellingQnetwork.parameters(), lr=0.005)
trainExplortionStrategy = epsilonGreedyAction(duellingQnetwork, 0.5, 0.05, 100)
evalExplortionStrategy = greedyAction(duellingQnetwork)


## choose wether prioritized replay buffer or uniform sampling replay buffer or implement your own
# replayBuffer = ExperienceReplayBuffer(bufferSize=10000) # uniform sampling, windowed memory
replayBuffer = PrioritizedExperienceRelpayBuffer(bufferSize=10000, alpha=0.6, beta=0.2, beta_rate=0.005) # prioritized sampling


# define the training strategy DQN in our example
agent = DDQN(env, duellingQnetwork, trainExplortionStrategy, optimizer, replayBuffer, 128, make_state=make_state,
                MaxTrainEpisodes=250, skipSteps=50, device=device, polyak_average=True, update_freq=1)
trainHistory = agent.trainAgent() 


# render
agent.evaluate(evalExplortionStrategy, EvalEpisodes=5, render=True)

# plots the training rewards v/s episodes
averaged_rewards = movingAverage(trainHistory['trainRewards'])
plt.plot([*range(len(trainHistory['trainRewards']))], averaged_rewards, label="train rewards")
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()
