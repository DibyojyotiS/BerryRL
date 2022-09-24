from typing import Tuple
from torch import Tensor

class ActionInhibitor:
    """The purpose of inhibitions on action """
    def __init__(self, dims:Tuple):
        pass

    def __call__(self, actions:Tensor):
        """This will scale the actions tensor with the
        corresponding inhibitions for each action.

        Parameters
        ----------
        action : Tensor
            _description_
        """