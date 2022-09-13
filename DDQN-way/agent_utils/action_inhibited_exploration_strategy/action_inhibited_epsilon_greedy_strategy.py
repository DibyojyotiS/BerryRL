from typing import List

import torch
import torch.nn.functional as F
from DRLagents.explorationStrategies import epsilonGreedyAction
from DRLagents.explorationStrategies.helper_funcs import entropy
from torch import nn, Tensor


class ActionInhibitedEpsilonGreedyActionStrategy(epsilonGreedyAction):
    def __init__(self, enable_episodes:List[int], epsilon=0.5, 
                    finalepsilon=None, decaySteps=None, 
                    decay_type='lin', outputs_LogProbs=False, 
                    print_args=False) -> None:
        """
        ### parameters
        - enable_episodes: List[int]
                - enables the i-th action after 
                the indicated number of episodes
        - epsilon: float (default 0.5) must be in [0,1)
        - finalepsilon: float (default None)
                - must be greater than 0
                - epsilon not decayed if finalepsilon is None
        - decaySteps: int (default None)
                - #calls to decay(...) in which epsilon decays to finalepsilon
                - if None, epsilon is not decayed
        - decay_type: str (default 'lin')
                - if 'lin' then epsilon decayed linearly
                - if 'exp' then epsilon decayer exponentialy
        - outputs_LogProbs: bool (default False)
                - if the model outputs the log-probablities
                - required for computing entropy for Policy Gradient methods
        - print_args: bool (default False)
                - print the agruments passed in init
        """
        super().__init__(
            epsilon=epsilon, finalepsilon=finalepsilon, 
            decaySteps=decaySteps, decay_type=decay_type, 
            outputs_LogProbs=outputs_LogProbs, print_args=print_args
        )
        self.enable_episodes = enable_episodes
        self.enabled_actions = [a for a,e in enumerate(enable_episodes) if e <= 0]
        self.disabled_actions = [a for a,e in enumerate(enable_episodes) if e > 0]
        assert len(self.enabled_actions), "atleast 1 action should be enabled"

    def decay(self):
        super().decay()
        for i in range(len(self.enable_episodes)):
            if self.enable_episodes[i] > 0:
                self.enable_episodes[i] -= 1
                if self.enable_episodes[i] <= 0:
                    self.enabled_actions.append(i)
                    self.disabled_actions = [
                        a for a,e in enumerate(self.enable_episodes) if e > 0
                    ]
                    self.enabled_actions_tensor = torch.tensor(
                        self.enabled_actions, device=self.device
                    )
        return

    def _lazy_init_details(self, model: nn.Module, state: Tensor):
        super()._lazy_init_details(model, state)
        self.enabled_actions_tensor = torch.tensor(
            self.enabled_actions, device=self.device
        )        
        return 

    def _epsilonGreedyActionUtil(self, model:nn.Module, state:Tensor, logProb_n_entropy:bool):

        sample = torch.rand(1)
        action_scores = None # dummy init action_scores
        if sample < self.epsilon:
            eGreedyAction = self._get_random_action()
        else:
            action_scores = model(state) # Q-values or action-log-probablities
            action_scores[:,self.disabled_actions] = float("-inf")
            eGreedyAction = torch.argmax(action_scores, dim=-1, keepdim=True)

        # if entropy and log-probablities are not required 
        if not logProb_n_entropy: return eGreedyAction

        # compute action_scores if not done
        if action_scores is None:
            action_scores = model(state)
        # if model outputs action-Q-values, convert them to log-probs
        if not self.outputs_LogProbs:
            log_probs = F.log_softmax(action_scores, dim=-1)
        else:
            log_probs = action_scores
        # compute the entropy and return log-prob of selected action
        _entropy = entropy(log_probs)

        return eGreedyAction, log_probs.gather(-1, eGreedyAction), _entropy

    def _get_random_action(self):
        n_enabled_actions = self.enabled_actions_tensor.shape[-1]
        rand_action_idx = torch.randint(n_enabled_actions, 
                                        size = (*self.output_shape[:-1],1), 
                                        device=self.device)
        eGreedyAction = self.enabled_actions_tensor[rand_action_idx]
        return eGreedyAction
