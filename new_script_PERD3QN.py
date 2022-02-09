import numpy as np
import torch
import torch.nn.functional as F
from berry_field.envs.berry_field_mat_input_env import BerryFieldEnv_MatInput
from berry_field.envs.utils.misc import getTrueAngles
from DRLagents import (DDQN, PrioritizedExperienceRelpayBuffer,
                       epsilonGreedyAction, greedyAction)
from torch import Tensor, nn
from torch.optim.rmsprop import RMSprop

from make_state import make_state

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


if __name__ == "__main__":
    # making the berry env
    berry_env = BerryFieldEnv_MatInput(no_action_r_threshold=0.6, 
                                        field_size=(5000,5000),
                                        initial_position=(2500,2500),
                                        penalize_boundary_hit=True)

    def env_reset(berry_env_reset):
        t=250
        n = 200
        def reset(**args):
            nonlocal t,n
            c = np.reshape(np.random.randint(2500-t,2500+t, size=2*n), (n,2))
            s = 10*np.random.randint(1,5, size=(n,1))
            berry_data = np.column_stack([s,c]).astype(float)
            x = berry_env_reset(berry_data=berry_data, initial_position=(2500,2500))
            berry_env.step(0)
            t=min(t+25, 2000)
            return x
        return reset
    berry_env.reset = env_reset(berry_env.reset)

    # init models
    value_net = make_net(3*8, 9, [16,8,8])
    buffer = PrioritizedExperienceRelpayBuffer(int(1E6), 0.9, 0.2, 0.001)

    # init optimizers
    optim = RMSprop(value_net.parameters(), lr=0.01)
    tstrat = epsilonGreedyAction(value_net, 0.25, 0.01, 50)
    estrat = greedyAction(value_net)

    agent = DDQN(berry_env, value_net, tstrat, optim, buffer, 128, gamma=0.99, 
                    skipSteps=20, make_state=make_state, printFreq=1,
                    snapshot_dir='.temp_stuffs/savesPERD3QN', device=TORCH_DEVICE)
    trianHist = agent.trainAgent(render=False)
    evalHist = agent.evaluate(estrat, 10, True)
