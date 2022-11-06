import numpy as np
from numba import njit, prange
from agent_utils.memories.base_class import MemoryBase


@njit
def njitUpdateMinTree(index, berryScore, minTree):
    memorySize = len(minTree)//2
    idx = index + memorySize
    minTree[idx][0] = berryScore
    minTree[idx][1] = index
    while idx > 1:
        idx//=2
        if minTree[2*idx][0] > minTree[2*idx+1][0]:
            minTree[idx] = minTree[2*idx+1]
        else:
            minTree[idx] = minTree[2*idx]


@njit
def njitRemoveBerriesOutOfRange(agentPosXY, minDistPopTh, maxDistPopTh, memory, minTree):
    distances = np.sum((memory[:,:-1] - np.asarray(agentPosXY))**2, axis=-1)**0.5
    indices = np.nonzero((distances < minDistPopTh) | (distances > maxDistPopTh))[0]
    for index in indices:
        memory[index][0] = memory[index][1] = memory[index][2] = -1
        njitUpdateMinTree(index, -1, minTree)


@njit
def njitUpdate(berryScore, berryPosXY, berrySize, agentPosXY, minDistPopTh, maxDistPopTh, memory, minTree):

    # remove any berry whose distance from agent is is out of range
    njitRemoveBerriesOutOfRange(agentPosXY, minDistPopTh, maxDistPopTh, memory, minTree)

    # add the new berry to the memory if it's score is greater than the minimum score
    dist = ((berryPosXY[0] - agentPosXY[0])**2 + (berryPosXY[1] - agentPosXY[1])**2)**0.5
    if (dist > minDistPopTh and dist < maxDistPopTh):
        min_berry_score = minTree[1][0]
        if berryScore > min_berry_score:
            min_index = int(minTree[1][1])
            memory[min_index] = [berryPosXY[0], berryPosXY[1], berrySize]
            njitUpdateMinTree(min_index, berryScore, minTree)


@njit
def njitBulkUpdata(listOfBerries, listOfberryScores, agentPosXY, minDistPopTh, maxDistPopTh, memory, minTree):
    for i in prange(len(listOfBerries)):
        berryScore = listOfberryScores[i]
        berryPosXY = listOfBerries[i, :2]
        berrySize = listOfBerries[i, 3]
        njitUpdate(berryScore, berryPosXY, berrySize, agentPosXY, minDistPopTh, maxDistPopTh, memory, minTree)


class NearbyBerryMemory(MemoryBase):
    def __init__(self, minDistPopTh, maxDistPopTh, memorySize) -> None:
        self.minDistPopTh = minDistPopTh
        self.maxDistPopTh = maxDistPopTh
        self.memorySize = memorySize

        self.__initMemory()

    def update(self, berryScore, berryPosXY, berrySize, agentPosXY):
        njitUpdate(berryScore, berryPosXY, berrySize, agentPosXY, self.minDistPopTh, self.maxDistPopTh, self.memory, self.minTree)

    def bulkUpdate(self, listOfBerries, listOfberryScores, agentPosXY):
        njitBulkUpdata(listOfBerries, listOfberryScores, agentPosXY, self.minDistPopTh, self.maxDistPopTh, self.memory, self.minTree)

    def getMemoryBerries(self):
        return NearbyBerryMemory.__filterMemory(self.memory)

    def reset(self,):
        self.__initMemory()

    def __initMemory(self):
        self.memory = np.zeros((self.memorySize, 3), dtype=np.float32) # x, y, berry_size
        self.minTree = np.zeros((2*self.memorySize, 2)) # berry_score, memory-index
        self.size = 0
        self.idx = 0

        # init min-tree
        for i in range(self.memorySize):
            njitUpdateMinTree(i, -1, self.minTree)

    @staticmethod
    @njit
    def __filterMemory(memory = np.array([[]])):
        indices = np.nonzero(memory[:,-1] > 0)[0]
        returnArray = np.zeros((len(indices), 3))
        for i, index in enumerate(indices):
            for j in range(3):
                returnArray[i][j] = memory[index][j]
        return returnArray


if __name__ == "__main__":
    from time import time_ns

    def temp_berry_worth(sizes:np.ndarray, distances:np.ndarray, 
                REWARD_RATE, DRAIN_RATE, HALFDIAGOBS, WORTH_OFFSET=0,
                min_berry_size=10, max_berry_size=40):
        rr, dr = REWARD_RATE, DRAIN_RATE
        worth = rr * sizes - dr * distances
        min_worth = rr * min_berry_size - dr * HALFDIAGOBS
        max_worth = rr * max_berry_size
        worth = (worth - min_worth)/(max_worth - min_worth)
        worth = np.clip(worth, a_min=0,a_max=None)
        worth = (worth + WORTH_OFFSET)/(1 + WORTH_OFFSET)
        return worth

    def setup():
        xy = np.random.randint(-1300, 1300, size=(80,2))
        sizes = np.random.randint(1,5, size=80)*10
        berries = np.column_stack([xy[:,0], xy[:,1], sizes])
        distances = np.linalg.norm(xy, axis=1)
        worths = temp_berry_worth(sizes, distances, REWARD_RATE=1e-4, DRAIN_RATE=1/(2*120*400), HALFDIAGOBS = 0.5*(1920**2 + 1080**2)**0.5)
        return berries, worths

    mem = NearbyBerryMemory(10, 150, 10)
    berries, worths = setup()
    mem.bulkUpdate(berries, worths, (0,0))
    mem.reset()

    times = []
    for i in range(10000):
        berries, worths = setup()
        startT = time_ns()
        mem.bulkUpdate(berries, worths, (0,0))
        times.append(time_ns() - startT)
    print("average time bulkUpdate:", np.average(times), "ns")