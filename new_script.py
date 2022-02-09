from berry_field.envs.berry_field_mat_input_env import BerryFieldEnv_MatInput
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from DRLagents import VPG, softMaxAction

from make_state import make_state

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_net(inDim, outDim, hDim, output_probs=False):
    class net(nn.Module):
        def __init__(self, inDim, outDim, hDim, activation = F.relu):
            super(net, self).__init__()
            self.inputlayer = nn.Linear(inDim, hDim[0])
            self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
            self.outputlayer = nn.Linear(hDim[-1], outDim)
            self.activation = activation

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            t = self.activation(self.inputlayer(x))
            for layer in self.hiddenlayers:
                t = self.activation(layer(t))
            t = self.outputlayer(t)
            if output_probs: t = F.log_softmax(t, -1)
            return t
    
    netw = net(inDim, outDim, hDim)
    netw.to(TORCH_DEVICE)
    return netw


if __name__ == "__main__":
    # making the berry env
    berry_env = BerryFieldEnv_MatInput(no_action_r_threshold=0.6)

    # init modelsS
    valuemodel = make_net(3*8, 1, [16,8])
    policymodel = make_net(3*8, 9, [16,8], output_probs=True)

    # init optimizers
    voptim = RMSprop(valuemodel.parameters(), lr=0.01)
    poptim = RMSprop(policymodel.parameters(), lr=0.01)
    tstrat = softMaxAction(policymodel, outputs_LogProbs=True)

    agent = VPG(berry_env, policymodel, valuemodel, tstrat, poptim, voptim, make_state, gamma=0.99,
                    MaxTrainEpisodes=500, MaxStepsPerEpisode=None, beta=0.1, value_steps=100,
                    trajectory_seg_length=2000, skipSteps=20, printFreq=1, device= TORCH_DEVICE,
                    snapshot_dir='.temp_stuffs/savesVPG')

    trianHist = agent.trainAgent(render=False)
    evalHist = agent.evaluate(tstrat, 10, True)