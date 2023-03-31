from typing import Any, Dict, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from berry_field import BerryFieldEnv
from DRLagents.agents.DDQN import greedyAction, softMaxAction
from debugging_utils.key_capture import KBHit
from config import BASE_CONFIG


class Debugger:
    def __init__(
        self,
        agent_config=Dict[str,Any],
        state_breakup:Dict[str,Tuple[Tuple[int,int],Tuple[int,int]]] = dict(
            sectorized_states = ((0, 32), (4,8)),
            edge_dists = ((32,36),(1,4)),
            misc_infos = ((36,43),(1,7)),
            time_memories = ((43,48), (1,5))
        ),
        plot_freq = 10,
        fig_size = (10,10),
        model_load_path = None
    ) -> None:
        self.state_breakup = state_breakup
        self.plot_freq = plot_freq
        self.torch_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        env = BerryFieldEnv()
        self.agent = Agent(
            berry_env_DRAIN_RATE=env.DRAIN_RATE,
            berry_env_FIELD_SIZE=env.FIELD_SIZE,
            berry_env_HALFDIAGOBS=env.HALFDIAGOBS,
            berry_env_REWARD_RATE=env.REWARD_RATE,
            torch_device=self.torch_device,
            **agent_config
        )
        self.env = self.agent.getPerceivedEnvironment(env)
        self.figure, self.axes = self._layout(state_breakup, fig_size)

        if model_load_path is not None:
            self.agent.nn_model.load_state_dict(
                torch.load(model_load_path)
            )

    def start(self):
        done = False
        action_selector = softMaxAction() #greedyAction()
        state = self.env.reset()
        num_steps = 0
        while not done:
            self.env.render()
            qvalues = self.agent.nn_model(
                torch.tensor(state, device=self.torch_device, dtype=torch.float32)
            )
            if num_steps % self.plot_freq == 0:
                self.show(qvalues.cpu().detach().numpy(), state)
            action = action_selector.select_action(qvalues).item()
            state, reward, done, info = self.env.step(action)
            num_steps += 1
        plt.show()

    def show(self, qvalues, state):
        self._matshow(qvalues[None,...], self.axes[0][0])
        self.axes[0][0].set_title("Qvalues")
        for i, k in enumerate(self.state_breakup.keys()):
            brk,shape = self.state_breakup[k]
            arr = np.reshape(state[brk[0]:brk[1]], shape)
            self._matshow(arr, self.axes[i+1][0])
            self.axes[i+1][0].set_title(k)
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        plt.pause(0.00000001)

    def _matshow(self, ndarray, ax:plt.Axes):
        ax.matshow(ndarray)
        for (i, j), z in np.ndenumerate(ndarray):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                    bbox=dict(
                        boxstyle='round', facecolor='white', edgecolor='0.3'
                    ), 
                    fontsize=8
                )

    def _layout(self, state_breakup:Dict[str,Any], fig_size:Tuple[int,int]):
        sizes = [(1,9), *[v[1] for k,v in state_breakup.items()]]
        fig, axes = plt.subplots(len(sizes), 1, squeeze=False, 
            gridspec_kw={
                'height_ratios': [1, *[x[0]/sizes[0][0] for x in sizes[1:]]], 
            },
            figsize=fig_size
        )
        return fig, axes

if __name__ == "__main__":
    debugger = Debugger(
        agent_config=BASE_CONFIG["AGENT"],
        model_load_path=".temp\\retrain\\0.01\\retrain-0.01 2022-12-6 3-35-36\\trainLogs\models\episode-594\onlinemodel_statedict.pt"
    )
    debugger.start()