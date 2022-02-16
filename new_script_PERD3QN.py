import pickle

import numpy as np
import torch
import torch.nn.functional as F
from berry_field.envs.berry_field_mat_input_env import BerryFieldEnv_MatInput
from DRLagents import (DDQN, PrioritizedExperienceRelpayBuffer,
                       epsilonGreedyAction, greedyAction)
from torch import nn
from torch.optim.rmsprop import RMSprop

from make_state import get_make_state, get_make_transitions
from mylogger import MyLogger


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

    save_dir = '.temp_stuffs/savesPERD3QN'
    print(save_dir)

    # setting up log file
    logger = MyLogger(save_dir+'/logout.txt',buff=1)

    # making the berry env
    buffer = PrioritizedExperienceRelpayBuffer(int(1E5), 0.95, 0.1, 0.001)
    berry_env = BerryFieldEnv_MatInput(no_action_r_threshold=float('inf'), 
                                        field_size=(4000,4000),
                                        initial_position=(2000,2000),
                                        observation_space_size=(1920,1080),
                                        end_on_boundary_hit= True,
                                        penalize_boundary_hit=True)

    def env_reset(berry_env_reset):
        t=200; n = 10; p = 5; r = 67
        episode_count = -1
        def reset(**args):
            nonlocal t, n, p, r, episode_count

            if episode_count>=0:
                print('episode:',episode_count, 
                    'berries picked:', berry_env.get_numBerriesPicked(),
                    'of', n*p, 'patches:', p, 'positive-in-buffer:',
                    sum(buffer.buffer['reward'].cpu()>0).item())

            # random berries
            patch_centroids = np.reshape(np.random.randint(400, 3600, size=2*p), (p,2))
            points = np.reshape(np.random.randint(-t,t, size=2*p*n), (n,p,2))
            berries = np.reshape(patch_centroids+points, (n*p,2))
            sizes = 10*np.random.randint(1,5, size=(n*p,1))
            berry_data = np.column_stack([sizes,berries]).astype(float)

            # compute an initial position
            rnd_idx = np.random.randint(0,n*p)
            rnd_angle = np.random.uniform(0,2*np.pi)
            initial_pos = [-1,-1]
            while any([(initial_pos[i]<20 or initial_pos[i]>=3980) for i in [0,1]]):
                initial_pos = berries[rnd_idx] + (np.random.randint(sizes[rnd_idx]+11,r)*\
                                np.array([np.cos(rnd_angle),np.sin(rnd_angle)])).astype(int)

            # reset the env  
            x = berry_env_reset(berry_data=berry_data, initial_position=initial_pos)

            episode_count+=1
            t= min(t+2, 300)
            # n= max(3, 10-2*(episode_count//100))
            # r= min(t, r+1)
            # p= max(3, 5 - 2*(episode_count//200))
            return x
        return reset

    def env_step(berry_env_step):
        print('no living cost: reward=(100*(reward>0)+(reward<=-1))*reward')
        def step(action):
            state, reward, done, info = berry_env_step(action)
            # reward = (10*(reward > 0) + (reward<=0))*reward
            reward = (100*(reward>0) + (reward<=-1))*reward # no living cost
            return state, reward, done, info
        return step

    berry_env.reset = env_reset(berry_env.reset)
    berry_env.step = env_step(berry_env.step)
    input_size, make_state_fn = get_make_state()
    make_transitions_fn = get_make_transitions(make_state_fn, look_back=100)

    # init models
    value_net = make_net(input_size, 9, [16,8,8])
    print(value_net)

    # init buffer and optimizers
    # buffer = PrioritizedExperienceRelpayBuffer(int(1E5), 0.95, 0.1, 0.001)
    optim = RMSprop(value_net.parameters(), lr=0.0001)
    tstrat = epsilonGreedyAction(value_net, 0.5, 0.01, 50)
    estrat = greedyAction(value_net)

    agent = DDQN(berry_env, value_net, tstrat, optim, buffer, 512, gamma=0.99, 
                    skipSteps=20, make_state=make_state_fn, make_transitions=make_transitions_fn, 
                    printFreq=1, update_freq=2, polyak_average=True, polyak_tau=0.2, 
                    snapshot_dir=save_dir, MaxTrainEpisodes=1000, device=TORCH_DEVICE)
    # trianHist = agent.trainAgent(render=True)
    trianHist = agent.trainAgent(render=False)

    with open(save_dir+'/history.pkl','wb') as f:
        pickle.dump(trianHist, f)