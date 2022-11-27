import numpy as np
from typing import Tuple
from .memories import MultiResolutionTimeMemory, NearbyBerriesMemory
from .memories import LocalityMemory

# TODO:
# 1. Allow disabling of a memory
#   - have functions like .hasNearbyBerryMemory() -> bool
#   - have an optional enabled flag in the ...MemoryKwarg parameters


class MemoryManager:
    def __init__(
        self, 
        berry_env_FIELD_SIZE:Tuple[int,int],
        multiResTimeMemoryKwargs = dict(
            grid_sizes=[(20,20),(50,50),(100,100),(200,200),(400,400)],
            factor=0.6, 
            exp=1.0,
        ),
        nearbyBerryMemoryKwargs = dict(
            minDistPopThXY=(1920/2, 1080/2), 
            maxDistPopThXY=(2600,2600), 
            memorySize=50
        ),
        localityMemoryKwargs = dict(
            resolution=(3,3)
        )
    ):
        self.berryEnvFIELD_SIZE = berry_env_FIELD_SIZE
        self.multiResTimeMemoryKwargs = multiResTimeMemoryKwargs
        self.nearbyBerriesMemoryKwargs = nearbyBerryMemoryKwargs
        self.localityMemoryKwargs = localityMemoryKwargs
        self.__initMemories()

    def update(
        self, recentlyPickedBerries:int, 
        listOfBerries:np.ndarray, 
        listOfBerryScores:np.ndarray, 
        agentPosXY:Tuple
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
        self.nearbyBerriesMemory.bulkInsertOrUpdate(
            listOfBerries, listOfBerryScores, agentPosXY
        )
        self.multiResTimeMemory.update(agentPosXY)
        self.berry_collected_count += recentlyPickedBerries
        self.localityMemory.update(agentPosXY)

    def reset(self):
        self.multiResTimeMemory.reset()
        self.nearbyBerriesMemory.reset()
        self.localityMemory.reset()
        self.berry_collected_count = 0

    def get_stats(self):
        return {
            "nearbyBerriesMemory":self.nearbyBerriesMemory.getStats()
        }

    # GETTERS >>>>>>>>>>>>>>>>>>
    def get_time_memories(self, agentPosXY):
        return self.multiResTimeMemory.get_time_memories(agentPosXY)

    def get_berry_memory(self, agentPosXY):
        return self.nearbyBerriesMemory.getMemoryBerriesAsRelativePos(agentPosXY)

    def get_num_berries_picked(self):
        return self.berry_collected_count

    def get_locality_memory(self):
        return self.localityMemory.get()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<

    def __initMemories(self):
        self.multiResTimeMemory = MultiResolutionTimeMemory(
            **self.multiResTimeMemoryKwargs,
            berryField_FIELD_SIZE=self.berryEnvFIELD_SIZE
        )
        self.nearbyBerriesMemory = NearbyBerriesMemory(
            **self.nearbyBerriesMemoryKwargs
        )
        self.localityMemory = LocalityMemory(
            **self.localityMemoryKwargs,
            berryfield_FIELD_SIZE=self.berryEnvFIELD_SIZE
        )
        self.berry_collected_count = 0