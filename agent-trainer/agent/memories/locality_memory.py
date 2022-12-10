import numpy as np
from typing import Tuple
from .base_class import MemoryBase

class LocalityMemory(MemoryBase):
    def __init__(
        self,
        berryfield_FIELD_SIZE: Tuple[int, int],
        resolution: Tuple[int, int]
    ) -> None:
        self.field_size = berryfield_FIELD_SIZE
        self.resolution = resolution
        self._initialize_localities()
        
    def _initialize_localities(self):
        self.size_x = self.field_size[0]//self.resolution[0]
        self.size_y = self.field_size[1]//self.resolution[1]
        self.localities_n_patches:np.ndarray = np.zeros((2,*self.resolution))

    def reset(self, *args, **kwargs):
        self.localities_n_patches[:] = 0

    def update(self, agent_pos_xy:Tuple[int, int], is_patch_seen:bool):
        x,y = agent_pos_xy
        if x == self.field_size[0]: x -= 1e-6
        if y == self.field_size[1]: y -= 1e-6
        
        locality_x = int(x/self.size_x)
        locality_y = int(y/self.size_y)
        self.localities_n_patches[0][locality_x][locality_y] = 1
        self.localities_n_patches[1][locality_x][locality_y] = is_patch_seen

    def get(self):
        return self.localities_n_patches