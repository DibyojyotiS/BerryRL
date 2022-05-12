# TRAIN AN AGENT IN A BABY ENVIRONMENT WITHOUT THE LIVING COST

import os
import pickle
import shutil
import time
import torch

from berry_field.envs.berry_field_env import BerryFieldEnv
from DRLagents import (DDQN, PrioritizedExperienceRelpayBuffer,
                       epsilonGreedyAction, greedyAction)
from DRLagents.utils.stdoutLogger import StdoutLogger

from torch.optim.rmsprop import RMSprop
from make_net import make_net

from make_state import get_make_state
from random_baby_env import random_baby_berryfield
# from make_state import get_make_transitions

BABY_FIELD_SIZE = (4000,4000)
PATCH_SIZE = (1000,1000)
NUM_PATCHES = 5
BERRIES_PER_PATCH = 10
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    save_dir = os.path.join('.temp_stuffs/w.o-negative-rewards/PERD3QN-state-with-action', 
                            '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6]))
    print(save_dir)

    # setting up log file
    logger = StdoutLogger(save_dir+'/logout.txt',buff=1)

    # copy make_state.py in save_dirs
    shutil.copy2('make_state.py', save_dir)
    shutil.copy2('make_net.py', save_dir)
    shutil.copy2('train_with_living_cost_PERD3QN.py', save_dir)

    # making the berry env
    random_berry_data, random_init_pos = random_baby_berryfield(BABY_FIELD_SIZE, PATCH_SIZE, 
                                                                NUM_PATCHES, BERRIES_PER_PATCH)
    berry_env = BerryFieldEnv(no_action_r_threshold=float('inf'),
                                        field_size=BABY_FIELD_SIZE,
                                        initial_position=random_init_pos,
                                        user_berry_data= random_berry_data,
                                        end_on_boundary_hit= True,
                                        penalize_boundary_hit=True)

    # redefine the reset function to generate random berry-envs
    buffer = PrioritizedExperienceRelpayBuffer(int(1E5), 0.95, 0.1, 0.01)
    def env_reset(berry_env_reset):
        episode_count = -1
        def reset(**args):
            nonlocal episode_count
            if episode_count>=0 and buffer.buffer is not None:
                print('-> episode:',episode_count, 'berries picked:', berry_env.get_numBerriesPicked(),
                    'of', NUM_PATCHES*BERRIES_PER_PATCH, 'patches:', NUM_PATCHES, 'positive-in-buffer:',
                    sum(buffer.buffer['reward'].cpu()>0).item())
            berry_data, initial_pos = random_baby_berryfield(BABY_FIELD_SIZE, PATCH_SIZE, 
                                                            NUM_PATCHES, BERRIES_PER_PATCH) # reset the env  
            x = berry_env_reset(berry_data=berry_data, initial_position=initial_pos)
            episode_count+=1
            return x
        return reset

    def env_step(berry_env_step):
        print('no living cost: reward=(100*(reward>0)+(reward<=-1))*reward')
        # print('all rewards scaled by 100 (except boundary hit)')
        def step(action):
            state, reward, done, info = berry_env_step(action)
            # if reward != -1: reward = 100*reward
            reward = (100*(reward>0) + (reward<=-1))*reward # no living cost
            return state, reward, done, info
        return step

    berry_env.reset = env_reset(berry_env.reset)
    berry_env.step = env_step(berry_env.step)
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
    trianHist = agent.trainAgent(render=False)

    with open(save_dir+'/history.pkl','wb') as f:
        pickle.dump(trianHist, f)