from OpenGL.GL import *
import glfw
import numpy as np

from . import constant_interface
from . import xtouch_interface


# TODO. Interface sometimes crashes for unknown reason.
# TODO. We want the y rotation to not depend on the x rotation.

class Visualizer:
  def __init__(self, interface='xtouch', defaults = {}):
    self.window_width = 1000
    self.window_height = 1000
    self.data = {}

    if not glfw.init():
      print("Failed to initialize OpenGL context")
    self.window = glfw.create_window(self.window_width, self.window_height, "visualizer", None, None)
    glfw.make_context_current(self.window)
    glPointSize(5)
    self.x_angle = 0
    self.y_angle = 0
    
    # Create interface.
    parameters = [
      ['x_angle', -180, 180],
      ['y_angle', -180, 180],
      ['scale', 0.1, 2],
    ]
    if interface == 'xtouch':
      parameters = [
        ['x_angle', -180, 180],
        ['y_angle', -180, 180],
        ['scale', 0.1, 2],
      ]
      self.interface = xtouch_interface.XTouchInterface(parameters)
    else:
      changes = {
        'x_angle': 1,
        'y_angle': 1,
      }
      self.interface = constant_interface.ConstantInterface(parameters, changes, defaults)

  def add(self, data):
    name = data['name']
    del data['name']
    # Pad if needed.
    data = data.copy()
    point_dim = data['points'].shape[1]
    if point_dim < 3:
      data['points'] = np.pad(data['points'], ((0,0), (0, 3-point_dim)), 'constant', constant_values=0)
    # Add to data.
    if name not in self.data: # Create new entry.
      data['points'] = data['points'].tolist()
      self.data.update({name: data})
    else: # Update existing entry.
      self.data[name]['points'] += data['points'].tolist()
      if data['color'] is not None:
        self.data[name]['color'] = data['color']

  def remove(self, name):
    self.data.pop(name, None)

  def render(self):
    self.get_mouse_pos()
    # Get rotations from interface. 
    interface_values = self.interface.get_values()
    self.x_angle = interface_values['x_angle']
    self.y_angle = interface_values['y_angle']
    self.scale = interface_values['scale']
    # Clear screen.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # Rotate.
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    # Rotations described in quaternions, with order z-axis, y-axis, x-axis (independent of glRotatef call order).
    glRotatef(self.y_angle, 0, 1, 0) 
    glRotatef(self.x_angle, 1, 0, 0) 
    glScalef(self.scale, self.scale, self.scale)
    # Draw.
    glBegin(GL_POINTS)
    for data in self.data.values():
      for point in data['points']:
        glColor3f(*data['color'])
        glVertex3f(*point)
    glEnd()
    # (Rotate.)
    glPopMatrix()
    # Render.
    glfw.swap_buffers(self.window)

  def add_rotation(self, x_angle, y_angle):
    self.x_angle += x_angle
    self.y_angle += y_angle

  def set_rotation(self, x_angle, y_angle):
    self.x_angle = x_angle
    self.y_angle = y_angle

  def set_scale(self, scale):
    self.scale = scale

  def get_mouse_pos(self): 
    x, y = glfw.get_cursor_pos(self.window)
    #v = glfw.get_cursor_pos(self.window)
    #import pdb; pdb.set_trace()
    y = self.window_height - y
    x = np.clip(x, 0, self.window_width)
    y = np.clip(y, 0, self.window_height)
    x = 2.0 * x / self.window_width - 1.0
    y = 2.0 * y / self.window_height - 1.0
    return x, y
