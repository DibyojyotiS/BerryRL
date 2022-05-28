import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from berry_field.envs.berry_field_env import BerryFieldEnv
from matplotlib.patches import Rectangle


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

def picture_episode(LOG_DIR, episode, K=10, figsize=(10,10)):
    # open the berry_field pickle and draw patches and berries
    path = f'{LOG_DIR}/analytics-berry-field/{episode}/berryenv.obj'
    with open(path, 'rb') as f: berry_field:BerryFieldEnv = pickle.load(f)
    berry_data = berry_field.berry_collision_tree.boxes
    patch_data = berry_field.patch_tree.boxes
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x=berry_data[:,0],y=berry_data[:,1],s=berry_data[:,2],c='red')
    ax.add_patch(Rectangle((0,0), *berry_field.FIELD_SIZE, fill=False))
    for x,y,pw,ph in patch_data: 
        ax.add_patch(Rectangle((x-pw/2,y-ph/2), pw, ph, fill=False))

    # open the agent path and draw the path with K fold decimation
    path = f'{LOG_DIR}/analytics-berry-field/{episode}/agent_path.txt'
    with open(path, 'r') as f: agentpath = eval('['+f.readline()+']')
    print('total steps', len(agentpath)-1)
    agentpath = np.array(agentpath[::K])
    ax.plot(agentpath[:,0],agentpath[:,1])
    plt.show()
