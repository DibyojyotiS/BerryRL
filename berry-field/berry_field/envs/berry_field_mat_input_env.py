import os
from typing import Tuple
from berry_field.envs.utils.misc import argsort_clockwise, getTrueAngles

import gym
import numpy as np
import copy
import pyglet
from gym.envs.classic_control import rendering

from .utils.collision_tree import collision_tree
from .utils.renderingViewer import renderingViewer


# location of the csv files with the location and description of each berry and patch
DATA_PATHS = ['data/berry_coordinates.csv', 'data/patch_coordinates.csv']
ABS_PATH = os.path.split(__file__)[0]

# env defaults that are really fixed defaults
MAX_DISPLAY_SIZE = (8*80, 4.5*80) # as (width, height)
FILE_PATHS = list(map(lambda x: os.path.join(ABS_PATH, x), DATA_PATHS))


class BerryFieldEnv_MatInput(gym.Env):
    def __init__(self,

                # environment defaults, all sizes are as (width, height)
                initial_position=(10000,10000), 
                field_size=(20000,20000), agent_size=10, observation_space_size=(1920, 1080),
                speed=400, # pixels per second
                maxtime=5*60, # default of 5 minutes max
                reward_rate=1e-4, 
                circular_berries=True, circular_agent=True,

                # can specify only your berries
                user_berry_data = None, # [size, x,y]

                # for curiosity reward
                reward_curiosity = False, reward_curiosity_beta=0.25,
                reward_grid_size = (100,100), # should divide respective dimention of field_size

                # announce picked berries with verbose
                verbose= False,

                # allow no action above this cumilaive reward
                no_action_r_threshold = -float('inf'),
                end_on_boundary_hit = False,
                penalize_boundary_hit = False,
                ):
        '''
        ## Environment\n
            all dimentions are in (width, height)\n
            initial_position: the initial position of the agent \n
            field_size: the size of field \n
            agent_size: the size of agent in (r,r) pixels \n
            observation_space_size: the visible portion \n
            speed: the speed of agent in pixels/sec \n
            maxtime: 5 minutes of max time \n
            reward_rate: berry size is scaled with this for reward \n
            circular_berries: wether berries are circular or square \n
            circular_agent: wether agetn is circular or square \n
            user_berry_data: can specify berries as list of [size, x,y]
                                overrides the original berry data


        ## curiosity reward\n
            reward_curiosity: wether curiosity reward is to be used\n
            reward_curiosity_beta: the reward is modified as - \n
                                    actual_reward*beta + curiosity_reward*(1-beta)\n
            reward_grid_size: break the field into smaller grids, agent gets\n
                                    rewarded on the first entry\n
        

        ## misc\n
            verbose: announce the picked berries in console\n
            no_action_r_threshold: doesnot allow action-0 (no-movement) when
                                    cumilative-reward is below this threshold
                    cumilative-reward is not incremented by curiosity-reward
            end_on_boundary_hit: end the episode on hitting the boundary
            penalize_boundary_hit: reward -1 on hitting the boundary
            
        '''
        super(BerryFieldEnv_MatInput, self).__init__()

        # Initializing variables
        self.FIELD_SIZE = field_size
        self.AGENT_SIZE = agent_size
        self.INITIAL_POSITION = initial_position
        self.DRAIN_RATE = 1/(2*120*speed)
        self.REWARD_RATE = reward_rate
        self.MAX_STEPS = speed*maxtime
        self.OBSERVATION_SPACE_SIZE = observation_space_size
        self.CIRCULAR_BERRIES = circular_berries
        self.CIRCULAR_AGENT = circular_agent
        self.END_ON_BOUNDARY_HIT = end_on_boundary_hit
        self.PENALIZE_BOUNDARY_HIT = penalize_boundary_hit

        # for the step machinary
        self.done = False
        self.position = initial_position
        self.num_steps = 0
        self.action_space = gym.spaces.Discrete(9)
        self.num_berry_collected = 0
        self.cummulative_reward = 0.5
        self.no_action_r_threshold = no_action_r_threshold

        self.action_switcher = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 1),
            3: (1, 0),
            4: (1, -1),
            5: (0, -1),
            6: (-1, -1),
            7: (-1, 0),
            8: (-1, 1)
        }

        # make the berry collision tree (in other words: populate the field)
        berry_data = self._read_csv(FILE_PATHS) if user_berry_data is None else user_berry_data 
        bounding_boxes, boxIds = self._create_bounding_boxes_and_Ids(berry_data)
        self.berry_radii = berry_data[:,0]/2 # [x,y,width,height]
        self.BERRY_COLLISION_TREE = collision_tree(bounding_boxes, boxIds, self.CIRCULAR_BERRIES, self.berry_radii) 
        self.berry_collision_tree = copy.deepcopy(self.BERRY_COLLISION_TREE) # only this copy will be modified during the runtime

        # announce picked berries with verbose
        self.verbose = verbose

        # for curiosity reward
        self.reward_curiosity = reward_curiosity
        self.reward_curiosity_beta = reward_curiosity_beta
        if reward_curiosity:
            assert all(f%z == 0 for f,z in zip(field_size, reward_grid_size))
            numx = field_size[0]//reward_grid_size[0]
            numy = field_size[1]//reward_grid_size[1]
            self.reward_grid_size = reward_grid_size
            self.size_visited_grid = (numx, numy)
            self.visited_grids = np.zeros((numx, numy))

        # for the rendering
        self.viewer = None
        self.current_action = 0


    def reset(self, info=False, initial_position=None, berry_data=None):
        """ info: bool, whether to return the info dict 
            initial_position: to change the initial position of agent
            berry_data: uses this berry data to reinit the env"""
        if self.viewer: self.viewer.close()
        
        self.done = False
        self.num_steps = 0
        self.viewer = None
        self.cummulative_reward = 0.5
        self.current_action = 0
        self.num_berry_collected = 0

        if initial_position is not None:
            self.position = initial_position
        else:
            self.position = self.INITIAL_POSITION

        if self.reward_curiosity:
            self.visited_grids = np.zeros(self.size_visited_grid)

        if berry_data is not None:
            bounding_boxes, boxIds = self._create_bounding_boxes_and_Ids(berry_data)
            self.berry_radii = berry_data[:,0]/2 # [x,y,width,height]
            self.berry_collision_tree = collision_tree(bounding_boxes, boxIds, self.CIRCULAR_BERRIES, self.berry_radii) 
        else:
            self.berry_collision_tree = copy.deepcopy(self.BERRY_COLLISION_TREE)

        if info:
            return self.raw_observation(), self.get_info()
        else:
            return self.raw_observation()


    def step(self, action):

        if self.no_action_r_threshold > self.cummulative_reward and action == 0:
            action = np.random.randint(0, 9)

        self.num_steps+=1
        self.current_action = action 

        # update the position
        movement = self.action_switcher[action]
        x = self.position[0] + movement[0]
        y = self.position[1] + movement[1]
        self.position = (  min(max(0, x), self.FIELD_SIZE[0]), 
                        min(max(0, y), self.FIELD_SIZE[1]) )

        # compute the reward
        juice_reward = self._pick_collided_berries()
        living_cost = - self.DRAIN_RATE*(action != 0)
        reward = juice_reward + living_cost
        self.cummulative_reward += reward

        # for curisity reward (don't add to self.cummulative_reward)
        if self.reward_curiosity:
            curiosity_reward = self.curiosity_reward()
            reward = reward * self.reward_curiosity_beta + \
                 curiosity_reward * (1 - self.reward_curiosity_beta)

        info = self.get_info()

        # did the episode just end?
        self.done = True if self.num_steps >= self.MAX_STEPS or \
                            self.cummulative_reward <= 0 or \
                            self.END_ON_BOUNDARY_HIT and self._has_hit_boundary() \
                         else False
        
        # -ve reward on hitting boundary with END_ON_BOUNDARY_HIT
        if self.PENALIZE_BOUNDARY_HIT and self._has_hit_boundary():
            reward = -1

        if self.done and self.viewer is not None: self.viewer = self.viewer.close()
        return self.raw_observation(), reward, self.done, info


    def get_info(self):

        x,y = self.position
        w,h = self.OBSERVATION_SPACE_SIZE
        W,H = self.FIELD_SIZE
        
        info = {
            # 'raw_observation': self.raw_observation(),
            'position':self.position,
            'cummulative_reward': self.cummulative_reward,
            'relative_coordinates': [self.position[0] - self.INITIAL_POSITION[0], 
                                     self.position[1] - self.INITIAL_POSITION[1]],
            'dist_from_edge':[
                w//2 - max(0, w//2 - x), # distance from left edge; w//2 if not in view
                w//2 - max(0, x+w//2 - W), # distance from right edge; w//2 if not in view
                h//2 - max(0, y+h//2 - H), # distance from the top edge; h//2 if not in view
                h//2 - max(0, h//2 - y) # distance from the bottom edge; h//2 if not in view
            ]
        }

        return info


    # returns visible berries as a list represented by their their center and sizes
    # make sure that self.position is correct/updated before calling this
    def raw_observation(self):
        ''' list of berries with center and size '''
        _, boxes = self._get_Ids_and_boxes_in_view((*self.position, *self.OBSERVATION_SPACE_SIZE))
        # compute distances and scale to 0-1
        boxes[:,:2] = boxes[:,:2] - self.position
        boxes[:, 0] = boxes[:, 0]/self.OBSERVATION_SPACE_SIZE[0]
        boxes[:, 1] = boxes[:, 1]/self.OBSERVATION_SPACE_SIZE[1]
        return boxes[:,:3]


    # the agent is rewarded by curiosity_reward when it enters a section for the first time 
    # make sure that self.position is correct/updated before calling this
    def curiosity_reward(self):
        x,y = self.position
        current_gridx = x//self.reward_grid_size[0]
        current_gridy = y//self.reward_grid_size[1]

        # for top and right boundaries
        if current_gridx >= self.size_visited_grid[0]: return 0
        if current_gridy >= self.size_visited_grid[1]: return 0

        curiosity_reward =  1 - self.visited_grids[current_gridx, current_gridy]
        self.visited_grids[current_gridx, current_gridy] = 1
        return curiosity_reward

    
    # pick the berries the agent collided with and return a reward
    # make sure that self.position is correct/updated before calling this
    def _pick_collided_berries(self):
        agent_bbox = (*self.position, self.AGENT_SIZE, self.AGENT_SIZE)
        boxIds, boxes = self.berry_collision_tree.find_collisions(agent_bbox, 
                                            self.CIRCULAR_AGENT, self.AGENT_SIZE/2, return_boxes=True)

        self.num_berry_collected += len(boxIds)
        if self.verbose and len(boxIds) > 0: print("picked ", len(boxIds), "berries")

        sizes = boxes[:,2] # boxes are an array with rows as [x,y, size, size]
        reward = self.REWARD_RATE * np.sum(sizes)
        self.berry_collision_tree.delete_boxes(list(boxIds))
        return reward


    def _get_Ids_and_boxes_in_view(self, bounding_box) -> Tuple[list, np.ndarray]:
        boxIds, boxes = self.berry_collision_tree.boxes_within_bound(bounding_box, return_boxes=True)
        return list(boxIds), boxes


    def _create_bounding_boxes_and_Ids(self, berry_data):
        """ bounding boxes from berry-coordinates and size """
        bounding_boxes = np.column_stack([
            berry_data[:,1:], berry_data[:,0], berry_data[:,0]
        ])
        boxIds = np.arange(bounding_boxes.shape[0])
        return bounding_boxes, boxIds


    def _read_csv(self, file_paths):
        # Constructing numpy arrays to store the coordinates of berries and patches
        berry_data = np.loadtxt(file_paths[0], delimiter=',') #[patch#, size, x,y]
        return berry_data[:, 1:] # [size,x,y]
    

    def render(self, mode="human"):

        assert mode in ["human", "rgb_array"]

        if self.done: 
            if self.viewer is not None: self.viewer = self.viewer.close()
            else: self.viewer = None
            return
        
        # berries in view
        screenw, screenh = self.OBSERVATION_SPACE_SIZE
        observation_bounding_box = (*self.position, screenw, screenh)
        agent_bbox = (screenw/2, screenh/2, self.AGENT_SIZE, self.AGENT_SIZE)
        boxIds, boxes = self._get_Ids_and_boxes_in_view(observation_bounding_box)
        boxes[:,0] -= self.position[0]-screenw/2; boxes[:,1] -= self.position[1]-screenh/2 
            
        # adjust for my screen size
        scale = min(1, min(MAX_DISPLAY_SIZE[0]/screenw, MAX_DISPLAY_SIZE[1]/screenh))
        screenw, screenh = int(screenw*scale), int(screenh*scale)
        if self.viewer is None: self.viewer = renderingViewer(screenw, screenh)
        self.viewer.transform.scale = (scale, scale)

        # draw berries
        if self.CIRCULAR_BERRIES:
            for center, radius in zip(boxes[:,:2], self.berry_radii[boxIds]):
                circle = rendering.make_circle(radius)
                circletrans = rendering.Transform(translation=center)
                circle.set_color(255,0,0)
                circle.add_attr(circletrans)
                self.viewer.add_onetime(circle)
        else:
            for x,y,width,height in boxes:
                l,r,b,t = -width/2, width/2, -height/2, height/2
                vertices = ((l,b), (l,t), (r,t), (r,b)) 
                box = rendering.FilledPolygon(vertices)
                boxtrans = rendering.Transform(translation=(x,y))
                box.set_color(255,0,0)
                box.add_attr(boxtrans)
                self.viewer.add_onetime(box)
        
        # draw agent
        if self.CIRCULAR_AGENT:
            agent = rendering.make_circle(self.AGENT_SIZE/2)
        else:
            p = self.AGENT_SIZE/2
            agentvertices =((-p,-p),(-p,p),(p,p),(p,-p))
            agent = rendering.FilledPolygon(agentvertices)
        agenttrans = rendering.Transform(translation=agent_bbox[:2])
        agent.set_color(0,0,0)
        agent.add_attr(agenttrans)
        self.viewer.add_onetime(agent)

        # draw boundary wall 
        l = observation_bounding_box[0] - observation_bounding_box[2]/2
        r = observation_bounding_box[0] + observation_bounding_box[2]/2 - self.FIELD_SIZE[0]
        b = observation_bounding_box[1] - observation_bounding_box[3]/2
        t = observation_bounding_box[1] + observation_bounding_box[3]/2 - self.FIELD_SIZE[1]
        top = observation_bounding_box[3] - t
        right = observation_bounding_box[2] - r
        if l<=0:
            line = rendering.Line(start=(-l, max(0, -b)), end=(-l,min(observation_bounding_box[2], top)))
            line.set_color(0,0,255)
            self.viewer.add_onetime(line)
        if r>=0:
            line = rendering.Line(start=(right, max(0, -b)), end=(right,min(observation_bounding_box[2], top)))
            line.set_color(0,0,255)
            self.viewer.add_onetime(line)
        if b<=0:
            line = rendering.Line(start=(max(0,-l), -b), end=(min(observation_bounding_box[2], right),-b))
            line.set_color(0,0,255)
            self.viewer.add_onetime(line)
        if t>=0:
            line = rendering.Line(start=(max(0,-l), top), end=(min(observation_bounding_box[1], right),top))
            line.set_color(0,0,255)
            self.viewer.add_onetime(line)

        # draw position and total reward
        label = pyglet.text.Label(f'x:{self.position[0]} y:{self.position[1]} a:{self.current_action} \t total-reward:{self.cummulative_reward:.4f} Step: {self.num_steps}', 
                                    x=screenw*0.1, y=screenh*0.9, color=(0, 0, 0, 255))
        self.viewer.add_onetimeText(label)

        return self.viewer.render(return_rgb_array=mode=="rgb_array")


    def get_numBerriesPicked(self):
        return self.num_berry_collected

    
    def _has_hit_boundary(self):
        return (self.position[0] == 0 or self.position[0]==self.FIELD_SIZE[0]) or \
               (self.position[1] == 0 or self.position[1]==self.FIELD_SIZE[1])