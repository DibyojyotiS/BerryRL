import pickle
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
from berry_field.envs.berry_field_env import BerryFieldEnv
from matplotlib.patches import Rectangle, Circle
import os
import imageio


def picture_episode(LOG_DIR:str, episode:int, K=10, figsize=(10,10), title=None, show=True, 
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
    pathlen = len(agentpath) # original length before decimation by K

    agentpath = np.array(agentpath[::K])
    ax.plot(agentpath[:,0],agentpath[:,1],linewidth=pathwidth)
    ax.axes.set_aspect('equal')

    if title: 
        nberries = ''
        try:
            path = f'{LOG_DIR}/analytics-berry-field/{episode}/results.txt'
            with open(path,'r') as f: 
                nberries = f.readlines()[-2].split(':')[-1].strip()
        except FileNotFoundError as ex: print(f'{path} not there!')
        plt.title(str(title)+f'\ntotal steps: {pathlen-1}' + f'\n{nberries} berries picked')
    if savepth: plt.savefig(savepth)
    if show: plt.show()
    if close: plt.close()


def _apply_kwargs(function, kwargs_dict):
    return function(**kwargs_dict)

def picture_episodes(fname:str, LOG_DIR:str, episodes:Iterable, K=10, figsize=(10,10), titlefmt='', 
                        alpha=1, pathwidth=1, duration=0.5, fps=1, nparallel=0, pretty=False):
    """ save the picture of episodes as .gif or as .mp4 depending on the fname """
    
    base_tmp_dir = os.path.join(LOG_DIR, ".tmp_pics")
    tmp_save_path_template = os.path.join(base_tmp_dir, 'temp_pic_episode_img_{}.png')
    if not os.path.exists(base_tmp_dir): os.makedirs(base_tmp_dir)

    def kwargs_gen(i):
        return dict(
            LOG_DIR=LOG_DIR, episode=i, K=K, 
            figsize=figsize, title=titlefmt.format(i), 
            show=False, alpha=alpha, pathwidth=pathwidth, 
            savepth=tmp_save_path_template.format(i), 
            close=True, pretty=pretty
        )

    if nparallel:
        import multiprocessing as mp
        with mp.Pool(nparallel) as pool: 
            pool.starmap(
                _apply_kwargs, 
                [(picture_episode, kwargs_gen(i)) for i in episodes]
            )

    if fname.endswith('.gif'):
        with imageio.get_writer(fname, duration=duration) as f:
            for i in episodes:
                if not nparallel: picture_episode(**kwargs_gen(i))
                img = imageio.imread(tmp_save_path_template.format(i))
                f.append_data(img)
                os.remove(tmp_save_path_template.format(i))

    if fname.endswith('.mp4'):
        with imageio.get_writer(fname, fps=fps) as f:
            for i in episodes:
                if not nparallel: picture_episode(**kwargs_gen(i))
                img = imageio.imread(tmp_save_path_template.format(i))
                f.append_data(img)
                os.remove(tmp_save_path_template.format(i))

    os.rmdir(base_tmp_dir)


# if __name__ == "__main__":
#     class foo:
#         def __init__(self,*args):
#             self.f = "Trained for {} episodes on random env\nEvaluation on fixed env\nEval-episode: {}"
#         def format(self,i,*args):
#             return self.f.format(10*i,i)
#     picture_episodes('evals.mp4','../eval',range(26),nparallel=12,titlefmt=foo())