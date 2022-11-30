from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .base_class import MemoryBase


# to visualize the memory up hill curve
def plot_time_mem_curves(factor,sizes,exp=1.0,berryfieldside=20000,n=1000):
    sizes = [max(size) for size in sizes]
    arr = [[0]*n for l in sizes]
    for i in range(n-1):
        for j,l in enumerate(sizes):
            delta = factor*l/berryfieldside
            arr[j][i+1] = arr[j][i]*(1-delta) + delta
    for j in range(len(sizes)):
        plt.plot(np.array(arr[j])**exp, label=f'{sizes[j]}->{berryfieldside//sizes[j]}')
    plt.grid()
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("magnitude")
    plt.show()


class MultiResolutionTimeMemory(MemoryBase):
    def __init__(
        self, 
        berryField_FIELD_SIZE: Tuple[int, int],
        grid_sizes: List[Tuple[int, int]],
        factor: float = 0.6,
        exp: float = 1.0,
        plot_curves=False
    ):
        """_summary_

        Parameters
        ----------
        berryField_FIELD_SIZE : Tuple[int, int]
            the size of the field
        grid_sizes : List[Tuple[int, int]]
            the berry-field is divided into (L,M) sized grid and the agent notes the time spent 
            in each of the cell. The memory of the time spent in a particular cell gets accessed 
            when the agent is in that cell.
        factor : float, optional
            increment the time of the current block
            by delta for each step in the block
            as an exponential average with (1-delta)
            where delta = time_memory_factor/resolution, by default 0.6
        exp : float, optional
            raise the stored time memory for the current block
            to time_memory_exp and feed to agent's state, by default 1.0
        """
        self.time_memory_grid_sizes = grid_sizes
        self.time_memory_factor = factor
        self.time_memory_exp = exp
        self.berryField_FIELD_SIZE = berryField_FIELD_SIZE
        self.cell_sizes = self._computeCellSizes()
        self.deltas = self._computeDecayDeltas(self.cell_sizes)
        self._setup()

        # some stuff that i need for feel
        if plot_curves:
            plot_time_mem_curves(self.time_memory_factor, 
                self.time_memory_grid_sizes, self.time_memory_exp, 
                self.berryField_FIELD_SIZE[0])
    
    def reset(self):
        self._reset_mem()
        self._reset_stats()

    def get_stats(self):
        return {
            "time_mem_max_stat": {
                f"grid-{size}": stat 
                for size,stat 
                in zip(self.time_memory_grid_sizes, self.time_mem_max_stat)
            }
        }

    def update(self, agentpos_XY):
        x = agentpos_XY[0]
        y = agentpos_XY[1]
        # handle edge case
        if x == self.berryField_FIELD_SIZE[0]: x -= 1e-6
        if y == self.berryField_FIELD_SIZE[1]: y -= 1e-6
        # decay time memory and update time_memory
        cell_pos = np.array([x,y]/self.cell_sizes, dtype=int)
        for i in range(len(self.time_memory_grid_sizes)):
            x_, y_ = cell_pos[i]
            self.time_mem_mats[i] *= 1-self.deltas[i]
            self.time_mem_mats[i][x_][y_] += self.deltas[i]
            self._update_stats(i,x_,y_)

    def get_time_memories(self, agentpos_XY):
        x = agentpos_XY[0]
        y = agentpos_XY[1]
        if x == self.berryField_FIELD_SIZE[0]: x -= 1e-6
        if y == self.berryField_FIELD_SIZE[1]: y -= 1e-6
        cell_pos = np.array([x,y]/self.cell_sizes, dtype=int)
        for i in range(len(self.time_mem_mats)):
            x_, y_ = cell_pos[i]
            self.current_time[i] = min(1,self.time_mem_mats[i][x_][y_])
        return self.current_time**self.time_memory_exp

    def _computeCellSizes(self):
        grid_sizes = np.array(self.time_memory_grid_sizes)
        cell_sizes = np.array(self.berryField_FIELD_SIZE)/grid_sizes
        return cell_sizes

    def _computeDecayDeltas(self, cell_sizes):
        deltas = self.time_memory_factor/np.max(cell_sizes,axis=1)
        return deltas

    def _setup(self):
        self.current_time = np.zeros(len(self.time_memory_grid_sizes))
        self.time_mem_mats = [
            np.zeros(shape) for shape in self.time_memory_grid_sizes
        ]

        self.time_mem_max_stat = np.zeros_like(self.current_time)
        
    def _reset_mem(self):
        for i in range(len(self.time_mem_mats)):
            self.time_mem_mats[i][:] = 0
        
    def _reset_stats(self):
        self.time_mem_max_stat[:] = 0

    def _update_stats(self, memory_index, x_, y_):
        self.time_mem_max_stat[memory_index] = max(
            self.time_mem_max_stat[memory_index],
            self.time_mem_mats[memory_index][x_][y_]
        )