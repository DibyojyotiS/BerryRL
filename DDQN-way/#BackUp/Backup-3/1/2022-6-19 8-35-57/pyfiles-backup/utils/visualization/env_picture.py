import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from berry_field.envs.berry_field_env import BerryFieldEnv
from matplotlib.patches import Rectangle, Circle
import os
import imageio


def picture_episode(LOG_DIR, episode, K=10, figsize=(10,10), title=None, show=True, 
                    alpha=1, pathwidth=1, savepth=None, close=False, pretty=False):
    """ plt plot showing patches, berries and the agent path """
    # open the berry_field pickle and draw patches and berries
    path = f'{LOG_DIR}/analytics-berry-field/{episode}/berryenv.obj'
    with open(path, 'rb') as f: berry_field:BerryFieldEnv = pickle.load(f)
    berry_data = berry_field.berry_collision_tree.boxes
    patch_data = berry_field.patch_tree.boxes
    fig, ax = plt.subplots(figsize=figsize)

    # scatter or put berries
    if pretty:
        ax.scatter(x=berry_data[:,0],y=berry_data[:,1],s=berry_data[:,2],c='red', alpha=alpha)
    else:
        for x,y,s,_ in berry_data:
            ax.add_patch(Circle((x,y),radius=s/2,color='r'))
    ax.add_patch(Rectangle((0,0), *berry_field.FIELD_SIZE, fill=False))
    for x,y,pw,ph in patch_data: 
        ax.add_patch(Rectangle((x-pw/2,y-ph/2), pw, ph, fill=False, alpha=alpha))

    # open the agent path and draw the path with K fold decimation
    path = f'{LOG_DIR}/analytics-berry-field/{episode}/agent_path.txt'
    with open(path, 'r') as f: agentpath = eval('['+f.readline()+']')
    print('total steps', len(agentpath)-1)
    agentpath = np.array(agentpath[::K])
    ax.plot(agentpath[:,0],agentpath[:,1],linewidth=pathwidth)
    ax.axes.set_aspect('equal')

    if title: 
        path = f'{LOG_DIR}/analytics-berry-field/{episode}/results.txt'
        with open(path,'r') as f: 
            nberries = f.readlines()[-2].split(':')[-1].strip()
        plt.title(str(title) + f'\n{nberries} berries picked')
    if savepth: plt.savefig(savepth)
    if show: plt.show()
    if close: plt.close()

def picture_episodes(fname, LOG_DIR, episodes, K=10, figsize=(10,10), titlefmt='', 
                        alpha=1, pathwidth=1, duration=0.5, fps=1, nparallel=0, pretty=False):
    """ save the picture of episodes as .gif or as .mp4 depending on the fname """
    
    if not os.path.exists('.tmp_pics'): os.makedirs('.tmp_pics')

    def argen(i):
        return LOG_DIR,i,K,figsize,titlefmt.format(i),False,\
            alpha,pathwidth,f'.tmp_pics/temp_pic_episode_img_{i}.png',True, pretty

    if nparallel:
        import multiprocessing as mp
        with mp.Pool(nparallel) as pool: 
            pool.starmap(picture_episode, [argen(i) for i in episodes])

    if fname.endswith('.gif'):
        with imageio.get_writer(fname, duration=duration) as f:
            for i in episodes:
                if not nparallel: picture_episode(*argen(i))
                img = imageio.imread(f'.tmp_pics/temp_pic_episode_img_{i}.png')
                f.append_data(img)
                os.remove(f'.tmp_pics/temp_pic_episode_img_{i}.png')

    if fname.endswith('.mp4'):
        with imageio.get_writer(fname, fps=fps) as f:
            for i in episodes:
                if not nparallel: picture_episode(*argen(i))
                img = imageio.imread(f'.tmp_pics/temp_pic_episode_img_{i}.png')
                f.append_data(img)
                os.remove(f'.tmp_pics/temp_pic_episode_img_{i}.png')

    os.removedirs('.tmp_pics')


# if __name__ == "__main__":
#     class foo:
#         def __init__(self,*args):
#             self.f = "Trained for {} on random env\nEvaluation on fixed env\nEval-episode: {}"
#         def format(self,i,*args):
#             return self.f.format(10*i,i)
#     picture_episodes('evals.mp4','../eval',range(26),nparallel=12,titlefmt=foo())