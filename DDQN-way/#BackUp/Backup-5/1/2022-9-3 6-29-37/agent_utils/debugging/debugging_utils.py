from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Circle
import numpy as np
from torch import float32, nn, tensor

EPSILON = 1e-8

def draw_berry_field(ax:Axes, agent, berries, 
                    OBSERVATION_SPACE_SIZE = (1920,1080), 
                    BERRY_FIELD_SIZE= (20000,20000)):
    """ 
    ### parameters:  
    - ax: a plt.Axes scalar object
    - agent: tupple[float, float, float]
            - (x,y,size)
    - berries: ndarray
            - [[x,y,size],...]
    - OBSERVATION_SPACE_SIZE: Tupple[float,foat] 
            - size of the observation space
            - default is (1920,1080)
     - BERRY_FIELD_SIZE: Tupple[float,foat]
            - size of the berry-field
            - default is (20000,20000)
    """
    W,H = BERRY_FIELD_SIZE
    w,h = OBSERVATION_SPACE_SIZE
    # plot the berry-field
    ax.add_patch(Rectangle((agent[0]-w/2, agent[1]-h/2), w,h, fill=False))
    ax.add_patch(Rectangle((agent[0]-w/2-30,agent[1]-h/2-30), w+60,h+60, fill=False))
    ax.scatter(x=berries[:,0], y=berries[:,1], s=berries[:,2], c='r')
    ax.scatter(x=agent[0], y=agent[1], s=agent[2], c='black')
    if agent[0]-w/2 < 0: ax.add_patch(Rectangle((0, agent[1] - h/2), 1, h, color='blue'))
    if agent[1]-h/2 < 0: ax.add_patch(Rectangle((agent[0] - w/2, 0), w, 1, color='blue'))
    if W-agent[0]-w/2<0: ax.add_patch(Rectangle((W, agent[1] - h/2), 1, h, color='blue'))
    if H-agent[1]-h/2<0: ax.add_patch(Rectangle((agent[0] - w/2, H), w, 1, color='blue'))

def draw_Q_values(ax:Axes, agentpos, state, model, device=None, action_names=None):
    """ visualize the action-values / probs
    Assumes that the first 8 outputs to be the 8 directional actions
    ### parameters
    - ax: a plt.Axes scalar object
    - model: an nn.Module called on the state
    """
    originalqvals = model(tensor([state], dtype=float32, 
                         device=device)).detach()[0].cpu().numpy()

    maxidx = np.argmax(originalqvals)
    ax.text(agentpos[0]+20, agentpos[1]+20, 
        f'q:{originalqvals[maxidx]:.2f}:{action_names[maxidx]}')

    # add action-advs circles
    colorqs = originalqvals[:8]
    colors = (colorqs-min(colorqs))/(max(colorqs)-min(colorqs)+EPSILON)
    for angle in range(0, 360, 45):
        rad = 2*np.pi * (angle/360)
        x,y = 100*np.sin(rad), 100*np.cos(rad)
        c = colors[angle//45]
        ax.add_patch(
            Circle((agentpos[0]+x, agentpos[1]+y), 
            20, color=(c,c,0,1)))

    # set title
    str_qvals = [f"{np.round(x,2):.2f}" for x in originalqvals.tolist()]
    meanings = [action_names[i]+' '*(len(qv)-len(action_names[i])) for i,qv in enumerate(str_qvals)]
    ax.set_title(f'env-record with q-vals plot\nqvals: {" ".join(str_qvals)}\n       {" ".join(meanings)}')
