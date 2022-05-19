import torch
import numpy as np

def printLocals(name, locals:dict):
    print(name,':')
    for k in locals: print('\t',k,':',locals[k])
    print('\n')

def my_print_fn(berry_env, buffer, tstrat, ssize=256):
    cpunptolist = lambda t: t.cpu().numpy().tolist()
    def print_fn():
        if buffer.buffer is not None:
            visited_patches = [p for p in berry_env.patch_visited.keys() if berry_env.patch_visited[p] > 0]
            positive_idx = np.argwhere((buffer.buffer['reward']>0).cpu().squeeze())
            a,count = torch.unique(buffer.buffer['action'][positive_idx], return_counts=True)

            sample = buffer.sample(ssize)[0]
            positive_idxsmp = np.argwhere((sample["reward"]>0).cpu().squeeze())
            asmp,countsmp = torch.unique(sample['action'][positive_idxsmp], return_counts=True)

            print('-> berries picked:', berry_env.get_numBerriesPicked(),
                'of', berry_env.get_totalBerries(), '| patches-visited:', visited_patches, 
                '| positive-in-buffer:', positive_idx.shape[-1],
                f'| amount-filled: {100*len(buffer)/buffer.bufferSize:.2f}%')
            print('\t| epsilon:', tstrat.epsilon)
            print('\t| action-stats: ', cpunptolist(a), cpunptolist(count))
            print(f'\t| approx positives in sample {ssize}: {positive_idxsmp.shape[-1]}')
            print(f'\t| approx action-dist in sample {ssize}: {cpunptolist(asmp)} {cpunptolist(countsmp)}')
    return print_fn