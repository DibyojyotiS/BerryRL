from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
    plt.show()


class MultiResolutionTimeMemory:
    def __init__(
        self, 
        time_memory_grid_sizes: List[Tuple[int, int]],
        berryField_FIELD_SIZE: Tuple[int, int],
        time_memory_factor: float = 0.6,
        time_memory_exp: float = 1.0,
        persistence: float=0.8,
    ):
        self.time_memory_grid_sizes = time_memory_grid_sizes
        self.time_memory_factor = time_memory_factor
        self.time_memory_exp = time_memory_exp
        self.persistence = persistence
        self.berryField_FIELD_SIZE = berryField_FIELD_SIZE

        self.cell_sizes = self._compute_cell_sizes()
        self.deltas = self._compute_decay_deltas(self.cell_sizes)
        self.reset()

        # some stuff that i need for feel
        plot_time_mem_curves(self.time_memory_factor, 
            self.time_memory_grid_sizes, self.time_memory_exp, 
            self.berryField_FIELD_SIZE[0])

    def _compute_cell_sizes(self):
        grid_sizes = np.array(self.time_memory_grid_sizes)
        cell_sizes = np.array(self.berryField_FIELD_SIZE)/grid_sizes
        return cell_sizes

    def _compute_decay_deltas(self, cell_sizes):
        deltas = self.time_memory_factor/np.max(cell_sizes,axis=1)
        return deltas

    def reset(self):
        self.time_mem_mats = [
            np.zeros(shape) for shape in self.time_memory_grid_sizes
        ]
        self.time_memories = np.zeros(len(self.time_mem_mats))

    def update(self, x, y):
        # decay time memory and update time_memory
        current_time = np.zeros_like(self.time_memories)
        cell_pos = np.array([x,y]/self.cell_sizes, dtype=int)
        for i in range(len(self.time_memory_grid_sizes)):
            x_, y_ = cell_pos[i]
            self.time_mem_mats[i] *= 1-self.deltas[i]
            self.time_mem_mats[i][x_][y_] += self.deltas[i]

            # clip to 1
            current_time[i] = min(1,self.time_mem_mats[i][x_][y_])
        
        self.time_memories = self.time_memories*self.persistence +\
                            (1-self.persistence)*(current_time**self.time_memory_exp)

    def get_time_memories(self):
        return self.time_memories
