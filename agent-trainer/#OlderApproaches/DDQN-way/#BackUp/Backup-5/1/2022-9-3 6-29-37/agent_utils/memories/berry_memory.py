import numpy as np

from agent_utils.memories.base_class import MemoryBase

class BerryMemory(MemoryBase):
    def __init__(self, grid_size) -> None:
        self.grid_size = grid_size
        pass

    def build_grid():
        pass

    def update(self,):
        pass

    def reset(self,):
        pass

# incorrect code:
# # update the berry memory
# mem_size = self.berry_memory_grid_size
# _x = int(x/(self.berryField.FIELD_SIZE[0]/mem_size[0]))
# _y = int(y/(self.berryField.FIELD_SIZE[1]/mem_size[1]))
# key = (_x,_y)
# avg_size = float(avg_worth*40) # since worth function is in 0-1 range

# if avg_size < 10: 
#     # in case we are revisiting and the berry 
#     self.berry_memory.pop(key, 0)
# else: self.berry_memory[key] = (
#     _x*self.berryField.FIELD_SIZE[0]/mem_size[0] + mem_size[0]/2,
#     _y*self.berryField.FIELD_SIZE[1]/mem_size[1] + mem_size[1]/2,
#     max(avg_size, self.berry_memory.get(key,(0,0,0))[-1]))