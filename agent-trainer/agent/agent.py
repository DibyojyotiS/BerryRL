from typing import Any, Dict, Tuple

from berry_field import BerryFieldEnv
from torch import device

from .complex_actions.explorative_actions import RandomExplorationAction
from .env_adapter import BerryFieldEnvAdapter
from .memory_manager import MemoryManager
from .reward_perception import RewardPerception
from .state_computation import StateComputation
from .nn_utils.simple_dueling_net import SimpleDuelingNet


class Agent:
    def __init__(
        self, 
        berry_env_FIELD_SIZE: Tuple[int, int],
        berry_env_HALFDIAGOBS: float,
        berry_env_REWARD_RATE: float,
        berry_env_DRAIN_RATE: float,
        torch_device: device,
        skip_steps = 10,
        memory_config = dict(
            multiResTimeMemoryKwargs = dict(
                enabled = True,
                grid_sizes = [(20,20),(50,50),(100,100),(200,200),(400,400)],
                factor=0.6, 
                exp=1.0,
            ),
            nearbyBerryMemoryKwargs = dict(
                enabled = True,
                minDistPopThXY=(1920/2, 1080/2), 
                maxDistPopThXY=(2600,2600), 
                memorySize=50
            )
        ),
        state_computation_config = dict(
            persistance=0.8, 
            sector_angle=45,
            berryworth_offset=0.05,
            normalizing_berry_count = 800,
            noise=0.05
        ),
        exploration_subroutine_config = dict(
            reward_discount_factor=1.0,
            max_steps=float('inf')
        ),
        reward_perception_config = dict(
            max_clip=2, min_clip=-0.04,
            scale=400
        ),
        nn_model_config = dict(
            layers=[32,16,16],
            lrelu_negative_slope=-0.01
        )
    ) -> None:
        # the stuff in here is common for both train and eval
        self.torch_device = torch_device
        self.skip_steps = skip_steps
        self._initDuellingModel(
            nn_model_config, state_computation_config, memory_config
        )
        self.memory_manager = MemoryManager(
            berry_env_FIELD_SIZE,
            **memory_config
        )
        self.state_computer = StateComputation(
            berry_env_HALFDIAGOBS,
            berry_env_REWARD_RATE,
            berry_env_DRAIN_RATE,
            self.memory_manager, 
            **state_computation_config
        )
        self.random_exploration_action = RandomExplorationAction(
            torch_model=self.nn_model,
            state_computer=self.state_computer,
            n_skip_steps=self.skip_steps,
            torch_device=torch_device,
            **exploration_subroutine_config
        )
        self.reward_perception = RewardPerception(
            memory_manager=self.memory_manager,
            **reward_perception_config
        )
        self.env_adapter = BerryFieldEnvAdapter(
            random_exploration_action= self.random_exploration_action,
            reward_perception=self.reward_perception,
            memory_manager=self.memory_manager,
            state_computation=self.state_computer,
            on_env_reset=self.reset_agent,
            skip_steps=self.skip_steps
        )

    def reset_agent(self):
        # must not reset env_adapter here due to calling of eval-init before
        # call to get stats
        self.memory_manager.reset()
        self.state_computer.reset()

    def get_stats(self):
        return {
            "env_adapter":self.env_adapter.get_stats(),
            "memory_manager": self.memory_manager.get_stats()
        }

    def getPerceivedEnvironment(self, berry_env: BerryFieldEnv):
        # modify the environment step and reset 
        self.env_adapter.create_adapter_for_env(berry_env)
        return berry_env

    def nnet(self):
        return self.nn_model

    def _initDuellingModel(
        self, nn_model_config:Dict[str,Any], 
        comp_conf:Dict[str,Any], mem_conf:Dict[str,Any]
    ):
        state_shape = StateComputation.get_output_shape(
            comp_conf=comp_conf, mem_conf=mem_conf
        ) # input shape
        self.nn_model = SimpleDuelingNet(
            in_features= state_shape[0],
            n_actions=9,
            **nn_model_config
        ).to(self.torch_device)
        print(self.nn_model)
        print(
            'total-params: ', 
            sum(
                p.numel() 
                for p in self.nn_model.parameters() if p.requires_grad
            )
        )
