# TRAIN AN AGENT IN A BABY ENVIRONMENT WITHOUT THE LIVING COST

import pickle
import shutil
import numpy as np
import torch

from berry_field.envs.berry_field_env import BerryFieldEnv
from DRLagents import (DDQN, PrioritizedExperienceRelpayBuffer,
                       epsilonGreedyAction, greedyAction)

from torch.optim.rmsprop import RMSprop
from make_net import make_net

from make_state import get_make_state
# from make_state import get_make_transitions
from mylogger import MyLogger


TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    save_dir = '.temp_stuffs/with-negative-rewards/PERD3QN-state-with-action'
    print(save_dir)

    # setting up log file
    logger = MyLogger(save_dir+'/logout.txt',buff=1)

    # copy make_state.py in save_dirs
    shutil.copy2('make_state.py', save_dir)
    shutil.copy2('make_net.py', save_dir)
    shutil.copy2('train_with_living_cost_PERD3QN.py', save_dir)

    # making the berry env
    buffer = PrioritizedExperienceRelpayBuffer(int(1E5), 0.95, 0.1, 0.01)
    berry_env = BerryFieldEnv(no_action_r_threshold=float('inf'), 
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

            if episode_count>=0 and buffer.buffer is not None:
                print('-> episode:',episode_count, 
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
        # print('no living cost: reward=(100*(reward>0)+(reward<=-1))*reward')
        print('all rewards scaled by 100')
        def step(action):
            state, reward, done, info = berry_env_step(action)
            # reward = (100*(reward>0) + (reward<=-1))*reward # no living cost
            return state, 100*reward, done, info
        return step

    berry_env.reset = env_reset(berry_env.reset)
    berry_env.step = env_step(berry_env.step)
    # input_size, make_state_fn = get_make_state()
    # make_transitions_fn = get_make_transitions(make_state_fn, look_back=100)
    input_size, make_state_fn, make_transitions_fn = get_make_state()

    # init models
    value_net = make_net(input_size, 9, [16,8,8])
    print(value_net)

    # init buffer and optimizers
    optim = RMSprop(value_net.parameters(), lr=0.0001)
    tstrat = epsilonGreedyAction(value_net, 0.5, 0.01, 50)
    estrat = greedyAction(value_net)

    agent = DDQN(berry_env, value_net, tstrat, optim, buffer, 512, gamma=0.99, 
                    skipSteps=20, make_state=make_state_fn, make_transitions=make_transitions_fn, 
                    printFreq=1, update_freq=2, polyak_average=True, polyak_tau=0.2, 
                    snapshot_dir=save_dir, MaxTrainEpisodes=500, device=TORCH_DEVICE)
    # trianHist = agent.trainAgent(render=True)
    trianHist = agent.trainAgent(render=False)

    with open(save_dir+'/history.pkl','wb') as f:
        pickle.dump(trianHist, f)