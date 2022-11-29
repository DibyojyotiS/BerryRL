import numpy as np
from typing import Tuple, Dict, Any
from .memories import MultiResolutionTimeMemory
from .memories import NearbyBerriesMemory
from .memories import LocalityMemory


# TODO:
# 1. Allow disabling of a memory
#   - have functions like .hasNearbyBerryMemory() -> bool
#   - have an optional enabled flag in the ...MemoryKwarg parameters


class MemoryManager:
    def __init__(
        self, 
        berry_env_FIELD_SIZE:Tuple[int,int],
        multiResTimeMemoryKwargs:Dict[str,Any] = dict(
            enabled = True,
            grid_sizes=[(20,20),(50,50),(100,100),(200,200),(400,400)],
            factor=0.6, 
            exp=1.0,
        ),
        nearbyBerryMemoryKwargs:Dict[str,Any] = dict(
            enabled = True,
            minDistPopThXY=(1920/2, 1080/2), 
            maxDistPopThXY=(2600,2600), 
            memorySize=50
        ),
        localityMemoryKwargs:Dict[str,Any] = dict(
            enabled = False,
            resolution = (5,5)
        )
    ):
        self.berryEnvFIELD_SIZE = berry_env_FIELD_SIZE
        self.multiResTimeMemoryKwargs = multiResTimeMemoryKwargs
        self.nearbyBerriesMemoryKwargs = nearbyBerryMemoryKwargs
        self.localityMemoryKwargs = localityMemoryKwargs
        self.__initMemories()

    def has_time_memories(self):
        return self._has_time_mem

    def has_berry_memory(self):
        return self._has_berry_mem

    def has_locality_memory(self):
        return self._has_locality_mem

    def update(
        self, recentlyPickedBerries:int, 
        listOfBerries:np.ndarray, 
        listOfBerryScores:np.ndarray, 
        agentPosXY:Tuple,
        isPatchSeen: bool
    ):
        """ Update all the memories

        Parameters
        ----------
        listOfBerries : np.ndarray
            2D array with row data [x,y,berry-size]
            x & y are relative to the agent's position
            NOTE: Stack the berries to update before the berries to insert
        listOfBerryScores : np.ndarray
            1D array with scores for each berry
        agentPosXY : Tuple
            position of the agent in the berry field
        """
        if self._has_time_mem: self.multiResTimeMemory.update(agentPosXY)
        if self._has_berry_mem:
            self.nearbyBerriesMemory.bulkInsertOrUpdate(
                listOfBerries, listOfBerryScores, agentPosXY
            )
        if self._has_locality_mem:
            self.localityMemory.update(agentPosXY, isPatchSeen)
        self.berry_collected_count += recentlyPickedBerries

    def reset(self):
        if self._has_time_mem:
            self.multiResTimeMemory.reset()
        if self._has_berry_mem:
            self.nearbyBerriesMemory.reset()
        if self._has_locality_mem:
            self.localityMemory.reset()
        self.berry_collected_count = 0

    def get_stats(self):
        return {
            "nearbyBerriesMemory":self.nearbyBerriesMemory.getStats()
        }

    # GETTERS >>>>>>>>>>>>>>>>>>
    def get_time_memories(self, agentPosXY):
        if not self._has_time_mem:
            raise RuntimeError(
                "MultiResolutionTimeMemory not enabled "
                +  "and get_time_memories called")
        return self.multiResTimeMemory.get_time_memories(agentPosXY)

    def get_berry_memory(self, agentPosXY):
        if not self._has_berry_mem:
            raise RuntimeError(
                "NearbyBerriesMemory not enabled and get_berry_memory called")
        return self.nearbyBerriesMemory.getMemoryBerriesAsRelativePos(agentPosXY)

    def get_locality_memory(self):
        if not self._has_locality_mem:
            raise RuntimeError(
                "LocalityMemory not enabled and get_locality_memory called")
        return self.localityMemory.get()

    def get_num_berries_picked(self):
        return self.berry_collected_count
    # <<<<<<<<<<<<<<<<<<<<<<<<<<

    def __initMemories(self):
        self._has_time_mem = self.__is_enabled(self.multiResTimeMemoryKwargs)
        self._has_berry_mem = self.__is_enabled(self.nearbyBerriesMemoryKwargs)
        self._has_locality_mem = self.__is_enabled(self.localityMemoryKwargs)

        if self._has_time_mem:
            self.multiResTimeMemory = MultiResolutionTimeMemory(
                **self.__remove_flag(self.multiResTimeMemoryKwargs),
                berryField_FIELD_SIZE=self.berryEnvFIELD_SIZE
            )

        if self._has_berry_mem:
            self.nearbyBerriesMemory = NearbyBerriesMemory(
                **self.__remove_flag(self.nearbyBerriesMemoryKwargs)
            )

        if self._has_locality_mem:
            self.localityMemory = LocalityMemory(
                **self.__remove_flag(self.localityMemoryKwargs),
                berryfield_FIELD_SIZE=self.berryEnvFIELD_SIZE
            )

        self.berry_collected_count = 0

    @staticmethod
    def __is_enabled(given_kwarg:Dict[str,Any]) -> bool:
        if given_kwarg is None:
            return False
        if "enabled" in given_kwarg:
            return given_kwarg["enabled"] is True
        return True

    @staticmethod
    def __remove_flag(given_kwarg:Dict[str,Any]) -> Dict[str,Any]:
        return {k:v for k,v in given_kwarg.items() if k != "enabled"}