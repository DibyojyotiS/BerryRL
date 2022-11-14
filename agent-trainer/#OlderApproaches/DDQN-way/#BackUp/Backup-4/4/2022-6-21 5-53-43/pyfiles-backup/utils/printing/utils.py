import torch
import numpy as np
from berry_field.envs import BerryFieldEnv

def printLocals(name, locals:dict):
    print(name,':')
    for k in locals: print('\t',k,':',locals[k])
    print('\n')

def Env_print_fn(berry_env:BerryFieldEnv):
    visited_patches = [p for p in berry_env.patch_visited.keys() \
                        if berry_env.patch_visited[p] > 0]
    print('-> berries picked:', berry_env.get_numBerriesPicked(),
        'of', berry_env.get_totalBerries(), '| patches-visited:', visited_patches,
        f'| juice left:{berry_env.total_juice:.2f}')
    
def my_print_fn(berry_env:BerryFieldEnv, buffer, tstrat, ddqntraininer):
    tolist = lambda t: t.cpu().numpy().tolist()
    def print_fn():
        ssize = ddqntraininer.batchSize
        try: skipsteps = ddqntraininer.skipSteps(ddqntraininer.current_episode)
        except Exception as e: skipsteps = ddqntraininer.skipSteps
        Env_print_fn(berry_env)
        print('\t| epsilon:', tstrat.epsilon)
        if buffer.buffer is not None:
            positive_idx = np.argwhere((buffer.buffer['reward']>0).cpu().squeeze())
            a,count = torch.unique(buffer.buffer['action'][positive_idx], return_counts=True)
            sample = buffer.sample(ssize)[0]
            positive_idxsmp = np.argwhere((sample["reward"]>0).cpu().squeeze())
            asmp,countsmp = torch.unique(sample['action'][positive_idxsmp], return_counts=True)
            # print the stuffs!
            print('\t| skipsteps:', skipsteps)
            print('\t| positive-in-buffer:', positive_idx.shape[-1],
                f'| amount-filled: {100*len(buffer)/buffer.bufferSize:.2f}%')
            print('\t| action-stats: ', tolist(a), tolist(count))
            print(f'\t| approx positives in sample {ssize}: {positive_idxsmp.shape[-1]}')
            print(f'\t| approx action-dist in sample {ssize}: {tolist(asmp)} {tolist(countsmp)}')
    return print_fn