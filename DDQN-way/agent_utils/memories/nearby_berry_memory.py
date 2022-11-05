import numpy as np
from numba import njit
from agent_utils.memories.base_class import MemoryBase


@njit
def updateMinTree(index, berry_score, min_tree):
    memory_size = len(min_tree)//2
    idx = index + memory_size
    min_tree[idx][0] = berry_score
    min_tree[idx][1] = index
    while idx > 1:
        idx//=2
        if min_tree[2*idx][0] > min_tree[2*idx+1][0]:
            min_tree[idx] = min_tree[2*idx+1]
        else:
            min_tree[idx] = min_tree[2*idx]


@njit
def removeBerriesOutOfRange(agent_pos_xy, max_dist_pop_th, min_dist_pop_th, memory, min_tree):
    distances = np.sum((memory[:,:-1] - np.asarray(agent_pos_xy))**2, axis=-1)**0.5
    indices = np.nonzero((distances > max_dist_pop_th) | (distances < min_dist_pop_th))[0]
    for index in indices:
        memory[index][0] = memory[index][1] = memory[index][2] = -1
        updateMinTree(index, -1, min_tree)


@njit
def njitupdate(berry_score, berry_pos_xy, berry_size, agent_pos_xy, max_dist_pop_th, min_dist_pop_th, memory, min_tree):

    # remove any berry whose distance from agent is is out of range
    removeBerriesOutOfRange(agent_pos_xy, max_dist_pop_th, min_dist_pop_th, memory, min_tree)

    # add the new berry to the memory if it's score is greater than the minimum score
    dist = ((berry_pos_xy[0] - agent_pos_xy[0])**2 + (berry_pos_xy[1] - agent_pos_xy[1])**2)**0.5
    if (dist > min_dist_pop_th and dist < max_dist_pop_th):
        min_berry_score = min_tree[1][0]
        if berry_score > min_berry_score:
            min_index = int(min_tree[1][1])
            memory[min_index] = [berry_pos_xy[0], berry_pos_xy[1], berry_size]
            updateMinTree(min_index, berry_score, min_tree)


class NearbyBerryMemory(MemoryBase):
    def __init__(self, dist_th1, dist_th2, memory_size) -> None:
        self.min_dist_pop_th = dist_th1
        self.max_dist_pop_th = dist_th2
        self.memory_size = memory_size

        self.__initMemory()

    def update(self, berry_score, berry_pos_xy, berry_size, agent_pos_xy):
        njitupdate(berry_score, berry_pos_xy, berry_size, agent_pos_xy, self.max_dist_pop_th, self.min_dist_pop_th, self.memory, self.min_tree)

    def getMemoryBerries(self):
        return NearbyBerryMemory.__filterMemory(self.memory)

    def reset(self,):
        self.__initMemory()

    def __initMemory(self):
        self.memory = np.zeros((self.memory_size, 3), dtype=np.float32) # x, y, berry_size
        self.min_tree = np.zeros((2*self.memory_size, 2)) # berry_score, memory-index
        self.size = 0
        self.idx = 0

        # init min-tree
        for i in range(self.memory_size):
            updateMinTree(i, -1, self.min_tree)

    @staticmethod
    @njit
    def __filterMemory(memory = np.array([[]])):
        indices = np.nonzero(memory[:,-1] > 0)[0]
        return_array = np.zeros((len(indices), 3))
        for i, index in enumerate(indices):
            for j in range(3):
                return_array[i][j] = memory[index][j]
        return return_array