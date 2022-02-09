
from berry_field.envs.berry_field_mat_input_env import BerryFieldEnv_MatInput
import torch
from make_state import make_state
from new_script import make_net
from DRLagents import (VPG, softMaxAction)
from torch.optim.rmsprop import RMSprop

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# making the berry env
berry_env = BerryFieldEnv_MatInput(no_action_r_threshold=0.6, verbose=True)

# init modelsS
valuemodel = make_net(3*8, 1, [16,8,8])
policymodel = make_net(3*8, 9, [16,8,8], output_probs=True)

# init optimizers
voptim = RMSprop(valuemodel.parameters(), lr=0.01)
poptim = RMSprop(policymodel.parameters(), lr=0.01)
tstrat = softMaxAction(policymodel, outputs_LogProbs=True)

valuemodel.load_state_dict(torch.load('.temp_stuffs\savesVPG\\value_model_weights_episode_3.pth'))
policymodel.load_state_dict(torch.load('.temp_stuffs\savesVPG\\policy_model_weights_episode_3.pth'))
valuemodel.eval()
policymodel.eval()

agent = VPG(berry_env, policymodel, valuemodel, tstrat, poptim, voptim, make_state, gamma=0.99,
                MaxTrainEpisodes=500, MaxStepsPerEpisode=None, beta=0.1, value_steps=10,
                trajectory_seg_length=2000, skipSteps=20, printFreq=1, device= TORCH_DEVICE)
evalHist = agent.evaluate(tstrat, 10, True)