from berry_field.envs.utils.misc import getTrueAngles
import numpy as np
from berry_field.envs.berry_field_mat_input_env import BerryFieldEnv_MatInput
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from DRLagents import DDQN, epsilonGreedyAction, greedyAction, PrioritizedExperienceRelpayBuffer

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make custom state using the env.step output
def make_state(list_raw_observation, info, angle = 45, kd=0.4, ks=0.1):
    # list_raw_observation a list of observations
    # raw_observations [x,y,size]
    raw_observation = list_raw_observation[-1]
    sizes = raw_observation[:,2]
    dist = np.linalg.norm(raw_observation[:,:2], axis=1, keepdims=True)
    directions = raw_observation[:,:2]/dist
    angles = getTrueAngles(directions)

    a1 = np.zeros(360//angle) # indicates sector with berries
    a2 = np.zeros_like(a1) # stores densities of each sector
    a3 = np.zeros_like(a1) # indicates the sector with the max worthy berry
    
    maxworth = float('-inf')
    maxworth_idx = -1
    for x in range(0,360,angle):
        sectorL = (x-angle/2)%360
        sectorR = (x+angle/2)
        if sectorL < sectorR:
            args = np.argwhere((angles>=sectorL)&(angles<=sectorR))
        else:
            args = np.argwhere((angles>=sectorL)|(angles<=sectorR))
        
        if args.shape[0] > 0: 
            idx = x//angle
            a1[idx] = 1
            # density of sector
            density = np.sum(sizes[args]**2)/1920*1080
            a2[idx] = density
            # max worthy
            worthyness = np.max(ks*sizes[args]-kd*dist[args])
            if worthyness > maxworth:
                maxworth_idx = idx
                maxworth = worthyness
    if maxworth_idx > -1: a3[maxworth_idx]=1 
    
    state = np.concatenate([a1,a2,a3])
    return state


def make_net(inDim, outDim, hDim, output_probs=False):
    class net(nn.Module):
        def __init__(self, inDim, outDim, hDim, activation = F.relu):
            super(net, self).__init__()
            self.outDim = outDim
            self.inputlayer = nn.Linear(inDim, hDim[0])
            self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
            self.outputlayer = nn.Linear(hDim[-1], outDim)
            self.activation = activation
            if outDim > 1 and not output_probs:
                self.actadvs = nn.Linear(hDim[-1], outDim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            t = self.activation(self.inputlayer(x))
            for layer in self.hiddenlayers:
                t = self.activation(layer(t))
            o = self.outputlayer(t) # value or probs
            if output_probs: o = F.log_softmax(o, -1)
            elif self.outDim > 1:
                advs = self.actadvs(t)
                o = o + (advs - advs.mean())
            return o
    
    netw = net(inDim, outDim, hDim)
    netw.to(TORCH_DEVICE)
    return netw



# making the berry env
berry_env = BerryFieldEnv_MatInput(no_action_r_threshold=0.6)

# init models
value_net = make_net(3*8, 9, [16,8])
buffer = PrioritizedExperienceRelpayBuffer(int(1E6), 0.9, 0.2, 0.001)

# init optimizers
optim = RMSprop(value_net.parameters(), lr=0.01)
tstrat = epsilonGreedyAction(value_net, 0.25, 0.01, 50)
estrat = greedyAction(value_net)

agent = DDQN(berry_env, value_net, tstrat, optim, buffer, 512, gamma=0.99, 
                skipSteps=50, make_state=make_state, printFreq=1,
                snapshot_dir='.temp_stuffs/saves', device=TORCH_DEVICE)
trianHist = agent.trainAgent(render=False)
evalHist = agent.evaluate(estrat, 10, True)