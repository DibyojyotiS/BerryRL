import pickle
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle


def picture_episode(
        berry_boxes:np.ndarray, patch_boxes:np.ndarray, sampled_path:np.ndarray, 
        nberries_picked:int, total_steps:int, field_size:Tuple[int,int],  
        figsize=(10,10), title=None, show=True, alpha=1, pathwidth=1, 
        savepth=None, close=False, pretty=False
):
    """ plt plot showing patches, berries and the agent path """
    # open the berry_field pickle and draw patches and berries
    fig, ax = plt.subplots(figsize=figsize)

    # scatter or put berries
    if pretty:
        ax.scatter(x=berry_boxes[:,0],y=berry_boxes[:,1],s=berry_boxes[:,2],c='red', alpha=alpha)
    else:
        for x,y,s,_ in berry_boxes:
            ax.add_patch(Circle((x,y),radius=s/2,color='r'))
    ax.add_patch(Rectangle((0,0), *field_size, fill=False))
    for x,y,pw,ph in patch_boxes: 
        ax.add_patch(Rectangle((x-pw/2,y-ph/2), pw, ph, fill=False, alpha=alpha))

    # visualize areas where the agent got stuck
    bin_x = field_size[0]/100
    bin_y = field_size[1]/100
    bins = [[bin_x*i for i in range(101)], [bin_y*i for i in range(101)]]
    hist = np.histogram2d(sampled_path[:,0], sampled_path[:,1], bins)[0]
    hist = np.clip(hist, 0, 3*(bin_x**2 + bin_y**2)**0.5)
    ax.matshow(-hist.transpose(), origin='lower', 
        extent=(0, field_size[0], 0, field_size[1]), 
        cmap='bone')

    # draw the path with K fold decimation
    ax.plot(sampled_path[:,0],sampled_path[:,1],linewidth=pathwidth)
    ax.axes.set_aspect('equal')

    if title: 
        plt.title(
            str(title) 
            + f'\ntotal steps: {total_steps}' 
            + f'\n{nberries_picked} berries picked'
        )
    if savepth: plt.savefig(savepth)
    if show: plt.show()
    if close: plt.close()


def juice_plot(
        sampled_juice:np.ndarray, total_steps:int, max_steps:int, title:str, 
        nberries_picked:int, figsize=(10,5), savepth=None, show=True, 
        close=False
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sampled_juice)
    ax.set_xlim(0, max_steps)
    ax.set_ylim(0,1)
    if title:
        plt.title(
            str(title) 
            + f'\ntotal steps: {total_steps}' 
            + f'\n{nberries_picked} berries picked'
        )
    if savepth: plt.savefig(savepth)
    if show: plt.show()
    if close: plt.close()

# def picture_episodes(fname:str, LOG_DIR:str, episodes:Iterable, K=10, figsize=(10,10), titlefmt='', 
#                         alpha=1, pathwidth=1, duration=0.5, fps=1, nparallel=0, pretty=False):
#     """ save the picture of episodes as .gif or as .mp4 depending on the fname """
    
#     if not os.path.exists('.tmp_pics'): os.makedirs('.tmp_pics')

#     def argen(i):
#         return LOG_DIR,i,K,figsize,titlefmt.format(i),False,\
#             alpha,pathwidth,f'.tmp_pics/temp_pic_episode_img_{i}.png',True, pretty

#     if nparallel:
#         import multiprocessing as mp
#         with mp.Pool(nparallel) as pool: 
#             pool.starmap(picture_episode, [argen(i) for i in episodes])

#     if fname.endswith('.gif'):
#         with imageio.get_writer(fname, duration=duration) as f:
#             for i in episodes:
#                 if not nparallel: picture_episode(*argen(i))
#                 img = imageio.imread(f'.tmp_pics/temp_pic_episode_img_{i}.png')
#                 f.append_data(img)
#                 os.remove(f'.tmp_pics/temp_pic_episode_img_{i}.png')

#     if fname.endswith('.mp4'):
#         with imageio.get_writer(fname, fps=fps) as f:
#             for i in episodes:
#                 if not nparallel: picture_episode(*argen(i))
#                 img = imageio.imread(f'.tmp_pics/temp_pic_episode_img_{i}.png')
#                 f.append_data(img)
#                 os.remove(f'.tmp_pics/temp_pic_episode_img_{i}.png')

#     os.removedirs('.tmp_pics')


# # if __name__ == "__main__":
# #     class foo:
# #         def __init__(self,*args):
# #             self.f = "Trained for {} episodes on random env\nEvaluation on fixed env\nEval-episode: {}"
# #         def format(self,i,*args):
# #             return self.f.format(10*i,i)
# #     picture_episodes('evals.mp4','../eval',range(26),nparallel=12,titlefmt=foo())