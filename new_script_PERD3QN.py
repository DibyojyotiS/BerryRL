import numpy as np
import torch
import torch.nn.functional as F
from berry_field.envs.berry_field_mat_input_env import BerryFieldEnv_MatInput
from DRLagents import (DDQN, PrioritizedExperienceRelpayBuffer,
                       epsilonGreedyAction, greedyAction)
from torch import Tensor, nn
from torch.optim.rmsprop import RMSprop

from make_state import get_make_state

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size, make_state_fn = get_make_state()

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
                                        reward_rate=0.001,
                                        field_size=(5000,5000),
                                        initial_position=(2500,2500),
                                        end_on_boundary_hit= True,
                                        penalize_boundary_hit=True)

    def env_reset(berry_env_reset):
        t=500
        n = 200
        def reset(**args):
            nonlocal t,n
            c = np.reshape(np.random.randint(2500-t,2500+t, size=2*n), (n,2))
            s = 10*np.random.randint(1,5, size=(n,1))
            berry_data = np.column_stack([s,c]).astype(float)
            x = berry_env_reset(berry_data=berry_data, initial_position=(2500,2500))
            berry_env.step(0)
            t=min(t+50, 2000)
            return x
        return reset

    def env_step(berry_env_step):
        def step(action):
            state, reward, done, info = berry_env_step(action)
            return state, 10*reward, done, info
        return step

    berry_env.reset = env_reset(berry_env.reset)
    berry_env.step = env_step(berry_env.step)

    # init models
    value_net = make_net(input_size, 9, [16,8,8])
    buffer = PrioritizedExperienceRelpayBuffer(int(1E3), 0.9, 0.2, 0.001)

    # init optimizers
    optim = RMSprop(value_net.parameters(), lr=0.01)
    tstrat = epsilonGreedyAction(value_net, 0.5, 0.01, 50)
    estrat = greedyAction(value_net)

    agent = DDQN(berry_env, value_net, tstrat, optim, buffer, 512, gamma=0.99, 
                    skipSteps=20, make_state=make_state_fn, printFreq=1, update_freq=2,
                    polyak_average=True, polyak_tau=0.2, snapshot_dir='.temp_stuffs/savesPERD3QN',
                    MaxTrainEpisodes=10, device=TORCH_DEVICE)
    trianHist = agent.trainAgent(render=False)
    evalHist = agent.evaluate(estrat, 1, True)
