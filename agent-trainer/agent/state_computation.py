from typing import List
import numpy as np

from .state_computation_utils import berry_worth, sectorized_states
from .memory_manager import MemoryManager


class StateComputation:
    def __init__(
        self, 
        berry_env_HALFDIAGOBS: float,
        berry_env_REWARD_RATE: float,
        berry_env_DRAIN_RATE: float,
        memory: MemoryManager, 
        persistance=0.8, 
        sector_angle=45,
        berryworth_offset=0.05,
        max_berry_count = 800,
        noise=0.05
    ) -> None: 
        """
        Computes the environment state using memories

        Parameters
        ----------
        memory : MemoryManager
        persistance : float, optional
            analogous to the persistence of vision
            used for some portion of the state, by default 0.8
        sector_angle : int, optional
            the observation space divided into angular sectors of this angle,
            by default 45
        berryworth_offset : float, optional
            offsets berry-worths by berryworth_offset/(1 + berryworth_offset)
            this amount., by default 0.05
        max_berry_count : int, optional
            normalize the #berries picked before adding 
            to the state, by default 800
        noise : float, optional
            uniform noise in [-noise, noise] to add to state, by default 0.05
        """
        self.berry_env_HALFDIAGOBS = berry_env_HALFDIAGOBS
        self.berry_env_REWARD_RATE = berry_env_REWARD_RATE
        self.berry_env_DRAIN_RATE = berry_env_DRAIN_RATE
        self.memory_manager = memory
        self.persistance = persistance
        self.angle = sector_angle
        self.worth_offset = berryworth_offset
        self.max_berry_count = max_berry_count
        self.noise = noise

        self.__init_compute()
        self.__init_constants()

    @staticmethod
    def get_output_shape(comp_conf:dict, mem_conf:dict):
        from berry_field.envs import BerryFieldEnv
        berry_env = BerryFieldEnv()
        state_comp = StateComputation(
            berry_env.HALFDIAGOBS, berry_env.REWARD_RATE, 
            berry_env.DRAIN_RATE, 
            MemoryManager(berry_env.FIELD_SIZE, **mem_conf),
            **comp_conf
        )
        obs = berry_env.reset()
        return state_comp.compute([[obs, 0, False, {}]], -1).shape

    def compute(self, skipTrajectory:List, actionTaken:int) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        skipTrajectory : List
            a list of [observation, info, reward, done]
            The observation is the list of berries visible to
            the agent
        actionTaken : int
            action repeated for the trajctory
        """
        observation = skipTrajectory[-1][0]
        listOfBerries = observation["berries"]
        position = observation['position']
        num_recentpicked = observation['num_berries_picked'] - self.berrycount

        # use the berries from memory to increase the field of view
        concated_berries = np.vstack([
            self.memory_manager.get_berry_memory(position), 
            listOfBerries
        ])

        berryworths, state = \
            self.__compute(observation, num_recentpicked, concated_berries)

        # update memories
        self.memory_manager.update(
            recentlyPickedBerries=num_recentpicked, 
            listOfBerries=concated_berries, 
            listOfBerryScores=berryworths, 
            agentPosXY=position,
            isPatchSeen=len(listOfBerries) > 0
        )
        self.berrycount += num_recentpicked

        uniformNoise = np.random.uniform(-self.noise, self.noise, size=state.shape)
        return state + uniformNoise

    def reset(self):
        self.__init_compute()

    def __init_compute(self):
        self.berrycount = 0
        self.prev_sectorized_state = None

    def __init_constants(self):
        self.ENV_HALFDIAG = self.berry_env_HALFDIAGOBS
        maxDistPopThXY = self.memory_manager.nearbyBerriesMemory.maxDistPopThXY
        self.BERRYMEM_MAXDIST = 0.5 * np.linalg.norm(maxDistPopThXY)

    def __berry_worth(self, berrySizes, berryDistances):
        return berry_worth(
            sizes=berrySizes,
            distances=berryDistances,
            REWARD_RATE=self.berry_env_REWARD_RATE,
            DRAIN_RATE=self.berry_env_DRAIN_RATE,
            HALFDIAGOBS=self.BERRYMEM_MAXDIST,
            WORTH_OFFSET=self.worth_offset
        )

    def __compute(self, observation, num_recentpicked, concated_berries):
        sectorizedState, _, berryworths = sectorized_states(
            listOfBerries=concated_berries,
            berry_worth_function=self.__berry_worth,
            maxPossibleDist=self.BERRYMEM_MAXDIST,
            prev_sectorized_state=self.prev_sectorized_state,
            persistence=self.persistance,
            angle=self.angle
        )
        self.prev_sectorized_state = sectorizedState

        state = np.concatenate([
            *sectorizedState,
            self.memory_manager.get_locality_memory().flatten(),
            self.memory_manager.get_time_memories(observation["position"]),
            observation['scaled_dist_from_edge'],
            [
                observation['patch_relative_score'],
                observation['total_juice'],
                len(observation["berries"])/50,
                num_recentpicked > 0, # bool picked feat
                min(1, self.berrycount/self.max_berry_count)
            ]
        ])
        
        return berryworths,state
