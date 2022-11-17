import logging
import pyglet

from berry_field.envs.field.field import Field

# setup rendering
try:
    from gym.envs.classic_control import rendering
    from berry_field.envs.rendering.renderingViewer import renderingViewer
    RENDERING_SETUP_SUCCESS = True
except Exception as ex:
    logging.warn("Failed to import gym.envs.classic_control.rendering "
                + f"due to exception {ex}")
    logging.warn("BerryFieldEnv.render(...) will not work and return None")
    RENDERING_SETUP_SUCCESS = False

MAX_DISPLAY_SIZE = (8*80, 4.5*80) # as (width, height)

class FieldRenderer:
    def __init__(self, field:Field, observation_space_size, field_size, agent_size) -> None:
        self.OBSERVATION_SPACE_SIZE = observation_space_size
        self.FIELD_SIZE = field_size
        self.AGENT_SIZE = agent_size
        self.field = field
        self.viewer = None
    
    def reset(self):
        if self.viewer is not None: self.viewer = self.viewer.close()
        self.viewer = None

    def render(
        self, done, position, current_action, total_juice, 
        num_steps, mode="human", circle_res=10
    ):
        if not RENDERING_SETUP_SUCCESS: return
        assert mode in ["human", "rgb_array"]
        if done: 
            self.reset()
            return

        # berries in view
        screenw, screenh = self.OBSERVATION_SPACE_SIZE
        observation_bounding_box = (*position, screenw, screenh)
        agent_bbox = (screenw/2, screenh/2, self.AGENT_SIZE, self.AGENT_SIZE)
        berry_boxes = self.field.get_berries_in_view(observation_bounding_box)
        berry_boxes[:,0] -= position[0]-screenw/2
        berry_boxes[:,1] -= position[1]-screenh/2 
            
        # adjust for my screen size
        scale = min(1, min(MAX_DISPLAY_SIZE[0]/screenw, MAX_DISPLAY_SIZE[1]/screenh))
        screenw, screenh = int(screenw*scale), int(screenh*scale)
        if self.viewer is None: self.viewer = renderingViewer(screenw, screenh)
        self.viewer.transform.scale = (scale, scale)

        # draw berries
        for (x,y,width,height) in berry_boxes:
            berry = rendering.make_circle(width/2, res=circle_res)
            translation = rendering.Transform(translation=(x,y))   
            berry.set_color(255,0,0)
            berry.add_attr(translation)
            self.viewer.add_onetime(berry)  
        
        # draw agent
        agent = rendering.make_circle(self.AGENT_SIZE/2, res=circle_res)
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
        label = pyglet.text.Label(
            f'x:{position[0]:.2f} y:{position[1]:.2f} a:{current_action} \
            \t total-reward:{total_juice:.4f} Step: {num_steps}', 
            x=screenw*0.1, y=screenh*0.9, color=(0, 0, 0, 255))
        self.viewer.add_onetimeText(label)

        return self.viewer.render(return_rgb_array=mode=="rgb_array")