from DRLagents import DDQN, PrioritizedExperienceRelpayBuffer, StdoutLogger

from get_baby_env import getBabyEnv
from make_net import *
from State_n_Transition_Maker import *

# baby env params
FIELD_SIZE = (4000,4000)
PATCH_SIZE = (1000,1000)
N_PATCHES = 5
N_BERRIES = 10

if __name__ == '__main__':
    berry_env = getBabyEnv(FIELD_SIZE, PATCH_SIZE, N_PATCHES, N_BERRIES)
    stMaker = State_n_Transition_Maker(berry_env)
    buffer = PrioritizedExperienceRelpayBuffer(int(1E5), 0.95, 0.1, 0.01)

    nnet = make_net(
        intDim = stMaker.get_output_shape(),
        outDim = berry_env.action_space.n,
        hDim = [64,64,64,64]
    )

    # an user-print-function to print extra stats
    def print_fn():
        if buffer.buffer is not None:
            visited_patches = [p for p in berry_env.patch_visited.keys() if berry_env.patch_visited[p] > 0]
            print('-> berries picked:', berry_env.get_numBerriesPicked(),
                'of', berry_env.get_totalBerries(), 'patches-visited:', visited_patches, 
                'positive-in-buffer:', sum(buffer.buffer['reward'].cpu()>0).item())

    ddqn_trainer = DDQN(berry_env, )

