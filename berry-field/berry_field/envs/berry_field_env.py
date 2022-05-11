import os
import pickle
import time
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


class BerryFieldEnv(gym.Env):
    def __init__(self,

                # environment defaults, all sizes are as (width, height)
                initial_position=(10000,10000), 
                field_size=(20000,20000), agent_size=10, observation_space_size=(1920, 1080),
                speed=400, # pixels per second
                maxtime=5*60, # default of 5 minutes max
                reward_rate=1e-4, 
                circular_berries=True, circular_agent=True,

                # can specify only your berries
                user_berry_data = None, # [patch-no, size, x,y]

                # for curiosity reward
                reward_curiosity = False, reward_curiosity_beta=0.25,
                reward_grid_size = (100,100), # should divide respective dimention of field_size

                # announce picked berries with verbose
                verbose= False,

                # allow no action above this cumilaive reward
                no_action_r_threshold = -float('inf'),
                end_on_boundary_hit = False,
                penalize_boundary_hit = False,

                # for analytics
                enable_analytics = True,
                analytics_folder = 'analytics-berry-field'
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
        super(BerryFieldEnv, self).__init__()

        # Initializing variables
        self.FIELD_SIZE = field_size
        self.AGENT_SIZE = agent_size
        self.INITIAL_POSITION = initial_position
        self.DRAIN_RATE = 1/(2*120*speed)
        self.MAXPOSDIST = 0.5*np.linalg.norm(observation_space_size) # 1/2 diagonal
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
        self.total_juice = 0.5
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


        # for the rendering
        self.viewer = None
        self.current_action = 0

        # announce picked berries with verbose
        self.verbose = verbose

        # init the structures (mainly collision trees)
        self.init_berryfield(user_berry_data)

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

        # for analytics
        self.analysis_enabled = enable_analytics
        self.analytics_folder = os.path.join(analytics_folder, '{}-{}-{} {}-{}-{}'.format(*time.gmtime()[0:6]))
        self.num_resets = 0 # used to index the analysis saves
        if enable_analytics: self.init_analysis(self.analytics_folder)

    
    def init_berryfield(self, user_berry_data):
        """ Inits the collision trees and other structures to make the field """
        # load and process the data
        berry_data = self._read_csv(FILE_PATHS) if user_berry_data is None else user_berry_data # [patch-no, size, x, y]
        berry_radii = berry_data[:,1]
        bounding_boxes = self._create_bounding_boxes(berry_data) # [x,y,width,height]
        patch_boxes = self._get_patch_boxes(berry_data) # compute patch boundaries using berry data assuming berries are already alloted to patches

        # make the berry collision tree (in other words: populate the field)
        self.ORIGINAL_BERRY_COLLISION_TREE = collision_tree(bounding_boxes, self.CIRCULAR_BERRIES, berry_radii) 

        # collision tree to detect the patch the agent is in
        self.ORIGINAL_PATCH_TREE = collision_tree(patch_boxes)

        # a look-up to get the patch for any berry
        self.BERRY_TO_PATCH_LOOKUP = berry_data[:,0].astype(int)

        # copies
        self.patch_tree = copy.deepcopy(self.ORIGINAL_PATCH_TREE)
        self.berry_collision_tree = copy.deepcopy(self.ORIGINAL_BERRY_COLLISION_TREE) # only this copy will be modified during the runtime
        self.patch_visited = {i:0 for i in range(len(self.patch_tree.boxes))}


    def reset_berryfield(self):
        """ resets the structures of berry-field to the original states """
        self.berry_collision_tree = copy.deepcopy(self.ORIGINAL_BERRY_COLLISION_TREE)
        self.patch_visited = {i:0 for i in range(len(self.patch_tree.boxes))}


    def init_analysis(self, save_folder):
        # create the save-folder to save analytics
        save_Folder = os.path.join(save_folder, f'{self.num_resets}')
        if not os.path.exists(save_Folder): os.makedirs(save_Folder)

        # save the data neccessary to rebuild the same berry-field
        with open(os.path.join(save_Folder, 'berryenv.obj'), 'wb') as f:
            pickle.dump(self, f,pickle.HIGHEST_PROTOCOL)



    def reset(self, info=False, initial_position=None, berry_data=None):
        """ info: bool, whether to return the info dict 
            initial_position: to change the initial position of agent
            berry_data: uses this berry data to reinit the env"""
        if self.viewer: self.viewer.close()
        
        self.done = False
        self.num_steps = 0
        self.viewer = None
        self.total_juice = 0.5
        self.current_action = 0
        self.num_berry_collected = 0
        
        # increment resets
        self.num_resets += 1

        self.position = initial_position if initial_position is not None else self.INITIAL_POSITION

        if self.reward_curiosity: self.visited_grids = np.zeros(self.size_visited_grid)

        if berry_data is not None: self.init_berryfield(berry_data)
        else: self.reset_berryfield()

        if self.analysis_enabled: self.init_analysis(self.analytics_folder)

        if info: return self.raw_observation(), self.get_info()
        else: return self.raw_observation()


    def step(self, action):

        assert not self.done

        if self.no_action_r_threshold > self.total_juice and action == 0:
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
        self.total_juice += reward

        # for curisity reward (don't add to self.cummulative_reward)
        if self.reward_curiosity:
            curiosity_reward = self.curiosity_reward()
            reward = reward * self.reward_curiosity_beta + \
                 curiosity_reward * (1 - self.reward_curiosity_beta)

        info = self.get_info()

        # did the episode just end?
        self.done = True if self.num_steps >= self.MAX_STEPS or \
                            self.total_juice <= 0 or \
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
        
        # scaled distance (x,y) relative to center of the patch the agent is in
        # if the agent is in no patch, then the all 1.0 is returned


        info = {
            'patch-relative':'',
            'position':self.position,
            'total_juice': self.total_juice,
            'relative_coordinates': [x - self.INITIAL_POSITION[0], 
                                     y - self.INITIAL_POSITION[1]],
            'scaled_dist_from_edge':[
                1 - max(0, w//2 - x)/(w//2), # scaled distance from left edge; 1 if not in view
                1 - max(0, x+w//2 - W)/(w//2), # scaled distance from right edge; 1 if not in view
                1 - max(0, y+h//2 - H)/(h//2), # scaled distance from the top edge; 1 if not in view
                1 - max(0, h//2 - y)/(h//2) # scaled distance from the bottom edge; 1 if not in view
            ]
        }

        return info

    def raw_observation(self):
        ''' returns visible berries as a list represented by their their center and sizes 
        make sure that self.position is correct/updated before calling this '''
        berry_boxes = self._get_berries_in_view((*self.position, *self.OBSERVATION_SPACE_SIZE))
        berry_boxes[:,:2] = berry_boxes[:,:2] - self.position
        # scale to 0-1
        berry_boxes[:, :2] = berry_boxes[:, :2]/self.MAXPOSDIST
        return berry_boxes[:,:3]


    def get_numBerriesPicked(self):
        return self.num_berry_collected


    def curiosity_reward(self):
        """ the agent is rewarded by curiosity_reward when it enters a section for the first time 
        make sure that self.position is correct/updated before calling this """
        x,y = self.position
        current_gridx = x//self.reward_grid_size[0]
        current_gridy = y//self.reward_grid_size[1]

        # for top and right boundaries
        if current_gridx >= self.size_visited_grid[0]: return 0
        if current_gridy >= self.size_visited_grid[1]: return 0

        curiosity_reward =  1 - self.visited_grids[current_gridx, current_gridy]
        self.visited_grids[current_gridx, current_gridy] = 1
        return curiosity_reward


    def render(self, mode="human", circle_res=10):

        assert mode in ["human", "rgb_array"]

        if self.done: 
            if self.viewer is not None: self.viewer = self.viewer.close()
            else: self.viewer = None
            return
        
        # berries in view
        screenw, screenh = self.OBSERVATION_SPACE_SIZE
        observation_bounding_box = (*self.position, screenw, screenh)
        agent_bbox = (screenw/2, screenh/2, self.AGENT_SIZE, self.AGENT_SIZE)
        berry_boxes, berry_ids = self._get_berries_in_view(observation_bounding_box, return_ids=True)
        berry_boxes[:,0] -= self.position[0]-screenw/2; berry_boxes[:,1] -= self.position[1]-screenh/2 
            
        # adjust for my screen size
        scale = min(1, min(MAX_DISPLAY_SIZE[0]/screenw, MAX_DISPLAY_SIZE[1]/screenh))
        screenw, screenh = int(screenw*scale), int(screenh*scale)
        if self.viewer is None: self.viewer = renderingViewer(screenw, screenh)
        self.viewer.transform.scale = (scale, scale)

        # draw berries
        for berryId, (x,y,width,height) in zip(berry_ids, berry_boxes):
            if self.CIRCULAR_BERRIES:
                berry = rendering.make_circle(width/2, res=circle_res)
            else:
                l,r,b,t = -width/2, width/2, -height/2, height/2
                vertices = ((l,b), (l,t), (r,t), (r,b)) 
                berry = rendering.FilledPolygon(vertices)
            translation = rendering.Transform(translation=(x,y))   
            berry.set_color(255,0,0)
            berry.add_attr(translation)
            self.viewer.add_onetime(berry)  
        
        # draw agent
        if self.CIRCULAR_AGENT:
            agent = rendering.make_circle(self.AGENT_SIZE/2, res=circle_res)
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
        label = pyglet.text.Label(f'x:{self.position[0]} y:{self.position[1]} a:{self.current_action} \t total-reward:{self.total_juice:.4f} Step: {self.num_steps}', 
                                    x=screenw*0.1, y=screenh*0.9, color=(0, 0, 0, 255))
        self.viewer.add_onetimeText(label)

        return self.viewer.render(return_rgb_array=mode=="rgb_array")

    
    def _pick_collided_berries(self):
        """pick the berries the agent collided with and return a reward
        make sure that self.position is correct/updated before calling this"""
        agent_bbox = (*self.position, self.AGENT_SIZE, self.AGENT_SIZE)
        boxIds, boxes = self.berry_collision_tree.find_collisions(agent_bbox, 
                                            self.CIRCULAR_AGENT, self.AGENT_SIZE/2, return_boxes=True)

        self.num_berry_collected += len(boxIds)
        if self.verbose and len(boxIds) > 0: print("picked ", len(boxIds), "berries")

        sizes = boxes[:,2] # boxes are an array with rows as [x,y, size, size]
        reward = self.REWARD_RATE * np.sum(sizes)
        self.berry_collision_tree.delete_boxes(list(boxIds))
        return reward


    def _get_patch_boxes(self, berry_data):
        """ generate the bounding boxes for the patches by taking the extreme berries
        it is assumed that berry data is of form [[patch-no., size, x, y],...] """

        # get the rectangular enclosure for all the patches
        num_patches = len(np.unique(berry_data[:,0]))
        patch_rects = [[np.inf,0.0,np.inf,0.0] for patch_no in range(num_patches)] #[left, right, bot, top]
        for berry in berry_data: # [patch-no, size, x, y]
            patch_id = int(berry[0])
            patch_rects[patch_id][0] = min(patch_rects[patch_id][0], berry[2]) # top x limit
            patch_rects[patch_id][1] = max(patch_rects[patch_id][1], berry[2]) # bot x limit
            patch_rects[patch_id][2] = min(patch_rects[patch_id][2], berry[3]) # top y limit
            patch_rects[patch_id][3] = max(patch_rects[patch_id][3], berry[3]) # bot y limit

        # convert rects to bounding boxes
        for i in range(num_patches):
            left, right, bot, top = patch_rects[i]
            centerx, centery = (left + right)/2, (bot + top)/2
            width = right - left
            height = top - bot
            patch_rects[i] = [centerx, centery, width, height]

        return np.array(patch_rects)


    def _get_berries_in_view(self, bounding_box, return_ids=False) -> Tuple[list, np.ndarray]:
        """ returns the bounding boxes of all the berries in the given bounding box """
        boxIds, boxes = self.berry_collision_tree.boxes_within_overlap(bounding_box, return_boxes=True)
        if return_ids: return boxes, boxIds
        return boxes


    def _create_bounding_boxes(self, berry_data):
        """ bounding boxes from berry-coordinates and size """
        bounding_boxes = np.column_stack([berry_data[:,2:], berry_data[:,1], berry_data[:,1]])
        return bounding_boxes


    def _read_csv(self, file_paths):
        """ Constructing numpy arrays to store the coordinates of berries and patches """
        berry_data = np.loadtxt(file_paths[0], delimiter=',') #[patch#, size, x,y]
        berry_data[:,0] -= min(berry_data[:,0]) # reindex patches from 0
        return berry_data # [patch-no,size,x,y]

    
    def _has_hit_boundary(self):
        return (self.position[0] == 0 or self.position[0]==self.FIELD_SIZE[0]) or \
               (self.position[1] == 0 or self.position[1]==self.FIELD_SIZE[1])