import os
from typing import Union
from berry_field.envs import BerryFieldEnv

import imageio
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from .debugging_utils import draw_Q_values, draw_berry_field

DEFAULT_ACTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 
                        'SW', 'W', 'NW', 'EX']

class Debugging:
    def __init__(self, debugDir:str, berryField:BerryFieldEnv) -> None:
        self.berryField = berryField
        self.OBSERVATION_SPACE_SIZE = berryField.OBSERVATION_SPACE_SIZE
        self.BERRY_FIELD_SIZE = berryField.FIELD_SIZE
        self.debugDir = os.path.join(debugDir, 'stMakerdebug')
        if not os.path.exists(self.debugDir): os.makedirs(self.debugDir)
        self.state_rec = open(f'{self.debugDir}/stMakerdebugstate.txt', 'w', 1)
        self.env_rec = open(f'{self.debugDir}/stMakerrecordenv.txt', 'w', 1)
        self.nlines = 0

    def _append_to_gif_file(self, giffile, fig):
        fig.savefig(f'{self.debugDir}/tmpimg.png')
        img = imageio.imread(f'{self.debugDir}/tmpimg.png')
        giffile.append_data(img)
        os.remove(f'{self.debugDir}/tmpimg.png')

    def _clear_axs(self,axs):
        for b in axs: 
            for a in b: a.clear() 

    def record(self, state:np.ndarray, **kwargs):
        agent, berries = self.berryField.get_human_observation()
        shape, l = state.shape, len(state.shape)
        np.savetxt(self.state_rec, [np.concatenate([[l],shape,state.flatten()])])
        np.savetxt(self.env_rec, [np.concatenate([agent, *berries])])
        self.nlines += 1

    def display_plot(self, plotfns, gif=False, figsize=(15,20), f=1):
        states = open(f'{self.debugDir}/stMakerdebugstate.txt','r')
        envdat = open(f'{self.debugDir}/stMakerrecordenv.txt','r')
        readline = lambda files: [f.readline() for f in files]
        line2arr = lambda l: np.array(eval('['+l[:-1].replace(' ',',')+']'),float)

        # init the gif file
        giffile = imageio.get_writer(
            f'{self.debugDir}/debug.gif') if gif else None


        # make the figure
        r,c = plotfns[0]
        fig,axs = plt.subplots(r,c, figsize=figsize,squeeze=False)
        plt.tight_layout(pad=5)

        for i in range(self.nlines):
            statel, envdtl = readline([states, envdat])
            if i%f != 0: continue
            
            # retrive the state and stuff from record
            state, envdt = line2arr(statel), line2arr(envdtl).reshape(-1,3)
            state = state[1+state[0].astype(int):].reshape(
                        state[1:1+state[0].astype(int)].astype(int))
            agentpos, berries = envdt[0], envdt[1:]

            # draw the berry-field
            draw_berry_field(axs[-1][-1], agentpos, berries, 
                self.OBSERVATION_SPACE_SIZE, self.BERRY_FIELD_SIZE)

            # draw the other plots
            for plotfn in plotfns: plotfn(axs, state, envdt)
            
            # append to gif and clear plots
            if giffile: self._append_to_gif_file(giffile,fig)
            plt.pause(0.001)
            self._clear_axs(axs)
        
        plt.show(); plt.close()
        states.close(); envdat.close()
        if giffile: giffile.close()

    def showDebug(self, plotfns=[(1,1)], nnet:Union[nn.Module,None]=None, 
            device=None, f=20, gif=False, action_names=None, figsize=(15,20)):
        """
        ### parameters
        - plotfns: list
            - the element at index zeros is the figure-size
            - the rest are functions that take in 
                - plt.Axes from plt.subplots(figure-size), 
                - the recorded state (a numpy.ndarray)
                - the env-record (a np.ndarray):
                    - the index-0 element is the agent position
                    - the rest are berries denoted as (x,y,size)
        - nnet: nn.Module
        - device: the device expected of the input to nnet
        - f: int (default 20) every f-th record is plotted
        - gif: bool (default False) make the gif 
        - action_names: 
        """
        actions = action_names if action_names else DEFAULT_ACTION_NAMES
    
        if nnet is not None:
            qvaluePlotter = lambda axs,state,envdt: draw_Q_values(axs[-1][-1],
                    agentpos=envdt[0], model=nnet, state=state, device=device,
                    action_names=actions)
            plotfns = plotfns + [qvaluePlotter]
        
        self.display_plot(plotfns=plotfns, gif=gif, figsize=figsize, f=f)
