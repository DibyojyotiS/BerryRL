from gym.envs.classic_control import rendering
import numpy as np
import pyglet
from pyglet.gl.gl import glClearColor

class renderingViewer(rendering.Viewer):
    def __init__(self, width, height, display=None):
        """ modified  gym.envs.classic_control.rendering to display text"""
        super(renderingViewer, self).__init__(width, height, display=display)
        self.onetime_texts = []
    
    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        for label in self.onetime_texts:
            label.draw()

        self.window.flip()
        self.onetime_geoms = []
        self.onetime_texts = []
        return arr if return_rgb_array else self.isopen

    def add_onetimeText(self, label):
        self.onetime_texts.append(label)