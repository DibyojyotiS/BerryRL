import os
from typing import Any, Dict, Tuple

import gym
import numpy as np
from berry_field.envs.analysis.analytics import BerryFieldAnalitics
from berry_field.envs.field.field import Field
from berry_field.envs.rendering.renderer import FieldRenderer


class BerryFieldEnv(gym.Env):

    INVSQRT2 = 0.5**0.5

    # csv files with the default data
    DATA_PATHS = ['data/berry_coordinates.csv', 'data/patch_coordinates.csv']
    FILE_PATHS = list(
        map(lambda x: os.path.join(os.path.split(__file__)[0], x), DATA_PATHS)
    )

    ACTION_SWITCHER = {
        0: (0, 1), # N
        1: (INVSQRT2, INVSQRT2), # NE
        2: (1, 0), # E
        3: (INVSQRT2, -INVSQRT2), # SE
        4: (0, -1), # S
        5: (-INVSQRT2, -INVSQRT2), # SW
        6: (-1, 0), # W
        7: (-INVSQRT2, INVSQRT2), # NW
    }

    def __init__(self,
        # environment defaults, all sizes are as (width, height)
        initial_position=(10000,10000), 
        field_size=(20000,20000), agent_size=10, observation_space_size=(1920, 1080),
        speed=400, # pixels per second
        maxtime=5*60, # default of 5 minutes max
        reward_rate=1e-4, 
        initial_juice = 0.5,

        # can specify only your berries
        berry_data = None, # [patch-no, size, x,y]

        # more stuffs
        end_on_boundary_hit = False,
        play_till_maxtime = False
    ):
        '''
        all dimentions are in (width, height)\n
        analytics are enabled always
        
        ## parameters

        ### ====== setup ======
        1. initial_position: tupple[int,int] (default (0,0)) 
                - the initial position of the agent
        2. field_size: tupple[int,int] (default (20000,20000)) 
                - the size of field
        3. agent_size: int (default 10) 
                - the diameter of agent in pixels
                - or the edge length if circular_agent is False
        4. observation_space_size:  tupple[int,int] (default (1920,1080))
                - the observable portion of the berry field
        5. speed: int (default 500)
                - the speed of agent in pixels/sec
        6. maxtime: int (default 300) 
                - the maximum game-time in seconds
                - 5 minutes is default
        7. reward_rate: float (default 1e-4)
                - berry size is scaled with this for reward
        10. berry_data: list|np.ndarray (default None)
                - can specify berries as list of [patch-no, size, x,y]
                - overrides the original berry data

        ### ====== misc ======
        19. end_on_boundary_hit: bool (default False) 
                - end the episode on hitting the boundary
        21. play_till_maxtime: bool (default False) 
                - if true lets the episode run untill maxtime
        '''
        super(BerryFieldEnv, self).__init__()

        # Initializing variables
        self.FIELD_SIZE = field_size
        self.AGENT_SIZE = agent_size
        self.INITIAL_POSITION = initial_position
        self.DRAIN_RATE = 1/(2*120*speed)
        self.HALFDIAGOBS = 0.5*np.linalg.norm(observation_space_size) # 1/2 diagonal
        self.REWARD_RATE = reward_rate
        self.MAX_STEPS = speed*maxtime
        self.OBSERVATION_SPACE_SIZE = observation_space_size
        self.END_ON_BOUNDARY_HIT = end_on_boundary_hit
        self.INTITAL_JUICE= initial_juice
        self.PLAY_TILL_MAXTIME= play_till_maxtime
        self.action_space = gym.spaces.Discrete(8)

        # init field
        self._default_init(
            agent_position=initial_position, berry_data=berry_data
        )

    def get_analysis(self):
        return self.analysis.get_analysis()

    def reset(self, initial_position=None, berry_data=None) -> Any:
        self._init_variables(initial_position)
        self.field.reset(berry_data=berry_data)
        self.renderer.reset()
        self.analysis.reset()
        observation, _, _ = self._get_observation_reward_info()
        self.done = self._is_done()
        return observation

    def render(self, mode="human", circle_res=10):
        self.renderer.render(
            done=self.done,
            position=self.position,
            current_action=self.current_action,
            total_juice=self.total_juice,
            num_steps=self.num_steps,
            mode=mode,
            circle_res=circle_res
        )

    def step(self, action: int) -> Tuple[Dict[str,Any], float, bool, Dict[str, Any]]:
        self.num_steps+=1
        self.current_action = action

        # update the position
        movement = self.ACTION_SWITCHER[action]
        x = self.position[0] + movement[0]
        y = self.position[1] + movement[1]
        self.position = (min(max(0, x), self.FIELD_SIZE[0]), 
                        min(max(0, y), self.FIELD_SIZE[1]))

        obs, reward, info = self._get_observation_reward_info()
        self.done = self._is_done()

        if self.done: self.renderer.reset()
        self.analysis.record_step(obs, reward, self.done, info)

        return obs, reward, self.done, info

    def _get_observation_reward_info(self):
        observationBB = (*self.position, *self.OBSERVATION_SPACE_SIZE)
        berry_boxes = self.field.get_berries_in_view(observationBB)
        berry_boxes[:,:2] = berry_boxes[:,:2] - self.position

        picked_berries = self.field.pick_collided_berries(self.position)
        juice_reward = np.sum(picked_berries[:,2])*self.REWARD_RATE
        reward = juice_reward - self.DRAIN_RATE
        
        patch_Id, patch_bb = self.field.get_current_patch(self.position)

        self.total_juice = max(0, self.total_juice + reward)
        self.num_berry_collected += len(picked_berries)

        observation = {
            "berries": berry_boxes[:,:3],
            "position": self.position,
            "scaled_dist_from_edge": self._get_scaled_dist_from_edge(),
            "recently_picked_berries": picked_berries[:,2],
            "patch_relative_score": self._get_patch_relative_score(patch_bb),
            "total_juice": self.total_juice,
        }

        info = {
            "current_patch_id": patch_Id,
            "num_berries_picked": self.num_berry_collected
        }

        return observation, reward, info
    
    def _get_patch_relative_score(self, current_patch_box):
        # scaled distance (x,y) relative to center of the patch the agent 
        # is in if the agent is in no patch, then the all 0.0 is returned 
        # (blends from patch to none)
        if current_patch_box is not None:
            x,y = self.position
            px,py,pw,ph = current_patch_box
            assert x >= px-pw/2, (x,y, px-pw/2, current_patch_box)
            assert y >= py-ph/2, (x,y, py-ph/2, current_patch_box)
            return min(1 - 2*abs(px-x)/pw, 1 - 2*abs(py-y)/ph)
        return 0.0

    def _get_scaled_dist_from_edge(self):
        x,y = self.position
        w,h = self.OBSERVATION_SPACE_SIZE
        W,H = self.FIELD_SIZE
        return [
            1 - max(0, w//2 - x)/(w//2), # from left edge; 1 if not in view
            1 - max(0, x+w//2 - W)/(w//2), # from right edge; 1 if not in view
            1 - max(0, y+h//2 - H)/(h//2), # from top edge; 1 if not in view
            1 - max(0, h//2 - y)/(h//2) # from bottom edge; 1 if not in view
        ]

    def _has_hit_boundary(self):
        return (self.position[0] == 0 or self.position[0]==self.FIELD_SIZE[0]) or \
               (self.position[1] == 0 or self.position[1]==self.FIELD_SIZE[1])

    def _is_done(self):
        # did the episode just end?
        done = True if self.num_steps >= self.MAX_STEPS or \
                    (self.total_juice <= 0 and not self.PLAY_TILL_MAXTIME) or \
                    (self.END_ON_BOUNDARY_HIT and self._has_hit_boundary()) \
                    else False
        return done

    def _init_variables(self, initial_position):
        self.done = False
        self.num_steps = 0
        self.num_berry_collected = 0
        self.current_action = None
        self.total_juice = self.INTITAL_JUICE
        if initial_position is None: initial_position = self.INITIAL_POSITION
        self.position = initial_position

    def _load_default_data(self, file_paths):
        """ Constructing numpy arrays to store the 
        coordinates of berries and patches """
        berry_data = np.loadtxt(file_paths[0], delimiter=',')
        berry_data[:,0] -= min(berry_data[:,0]) # reindex patches from 0
        return berry_data # [patch-no,size,x,y]

    def _default_init(self, agent_position, berry_data):
        if berry_data is None:
            berry_data = self._load_default_data(file_paths=self.FILE_PATHS)
        self.field = Field(agent_size=self.AGENT_SIZE, berry_data=berry_data)
        self.renderer = FieldRenderer(
            field=self.field,
            observation_space_size=self.OBSERVATION_SPACE_SIZE,
            field_size=self.FIELD_SIZE,
            agent_size=self.AGENT_SIZE
        )
        self.analysis = BerryFieldAnalitics(
            field=self.field,
            max_steps=self.MAX_STEPS,
            ignore_nresets=1
        )
        self._init_variables(agent_position)