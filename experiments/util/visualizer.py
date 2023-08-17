from OpenGL.GL import *
import glfw
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rotation 

from .constant_interface import ConstantInterface
from .xtouch_interface import XTouchInterface


# TODO. Interface sometimes crashes for unknown reason.
# TODO. We want the y rotation to not depend on the x rotation.
# TODO. Optimize this by using a buffer. 

class Visualizer:
  def __init__(self, interface='xtouch', defaults = {}, manifold=None, distance_function=None, cursor_target=None, cursor_color=None):
    self.window_width = 1000
    self.window_height = 1000
    self.max_points = 10000
    self.manifold = manifold
    #self.distance = lambda x, y: manifold.distance_function(manifold, x, y) if distance is None and manifold is not None else distance
    self.cursor_target = cursor_target
    self.cursor_color = cursor_color
    self.distance_function = distance_function 
    self.data = {}

    if not glfw.init():
      print("Failed to initialize OpenGL context")
      exit(1)
    
    self.window = glfw.create_window(self.window_width, self.window_height, "visualizer", None, None)
    
    if not self.window:
      print("Failed to create GLFW window")
      exit(1)

    glfw.make_context_current(self.window)
    glPointSize(5)
    
    self.x_angle = 0
    self.y_angle = 0
    self.scale = 0
    
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
      self.interface = XTouchInterface(parameters)
    else:
      changes = {
        'x_angle': 1,
        'y_angle': 1,
      }
      self.interface = ConstantInterface(parameters, changes, defaults)

  def add(self, data):
    name = data['name']
    del data['name']
    # Pad if needed.
    data = data.copy()
    n_points, point_dim = data['points'].shape
    if point_dim < 3:
      data['points'] = np.pad(data['points'], ((0,0), (0, 3-point_dim)), 'constant', constant_values=0)
    # Add to data.
    if name not in self.data: # Create new entry.
      self.data[name] = {
        'points': np.zeros([self.max_points, 3]),
        'n_points': n_points,
        'color': data['color']
      }
      self.data[name]['points'][:n_points] = data['points'][:]
    else: # Update existing entry.
      n_existing_points = self.data[name]['n_points']
      if n_existing_points + n_points <= self.max_points:
        self.data[name]['points'][n_existing_points : n_existing_points + n_points] = data['points'][:]
        self.data[name]['n_points'] += n_points
      else: # Randomly remove some existing points.
        np.random.shuffle(self.data[name]['points'])
        self.data[name]['points'][-n_points:] = data['points'][:]
        self.data[name]['n_points'] = self.max_points
      if data['color'] is not None: # Update color.
        self.data[name]['color'] = data['color']

  def remove(self, name):
    self.data.pop(name, None)

  def render(self):
    glfw.poll_events()
    if glfw.window_should_close(self.window):
      glfw.terminate()
      exit(0)
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
    #glRotatef(self.y_angle, 0, 1, 0) 
    #glRotatef(self.x_angle, 1, 0, 0) 
    #glScalef(self.scale, self.scale, self.scale)
    # Draw.
    transformation_matrix = self.scale * scipy_rotation.from_euler('yx', [self.y_angle, self.x_angle], degrees=True).as_matrix()
    sample_colors = None
    glBegin(GL_POINTS)
    for name in self.data:
      data = self.data[name]
      n_points = data['n_points']
      points = np.matmul(data['points'][:n_points], transformation_matrix)
      colors = None
      if self.cursor_target is not None and self.distance_function is not None and self.cursor_target == name:
        colors = self.compute_colors(points, data['color']).tolist()
      for i, point in enumerate(points): 
        color = data['color'] if colors is None else colors[i] 
        glColor3f(*color)
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

  def closest_point_to_mouse(self, points):
    x, y = self.get_mouse_pos()
    closest_point = None
    closest_distance = np.inf
    for point in points: 
      distance = np.linalg.norm(np.array([x, y]) - np.array(point[:2]))
      if distance < closest_distance:
        closest_distance = distance
        closest_point = point
    return closest_point

  def compute_colors(self, points, color):
    closest_point = self.closest_point_to_mouse(points)
    assert closest_point is not None
    colors = np.zeros([points.shape[0], 3])
    # TODO. Should be parallelizable.
    for i, point in enumerate(points):
      distance = self.distance_function(point, closest_point) / self.scale # We must adjust for the scale.
      color_scaling = np.exp(3.0 * -distance)
      #color_scaling = np.max([0, 1.0 - distance * 3.0])
      #colors[i] = [color[0], color[1], 255 * color_scaling]
      colors[i] = self.interpolate_color(self.cursor_color, color, color_scaling)
    return colors

  def render_closest_point(self, points, color):
    x, y = self.get_mouse_pos()
    closest_point = None
    closest_distance = np.inf
    for point in points: 
      #distance = self.manifold.distance(mouse_projected, point)
      distance = np.linalg.norm(np.array([x, y]) - np.array(point[:2]))
      if distance < closest_distance:
        closest_distance = distance
        closest_point = point
    if closest_point is not None:
      glPointSize(20)
      glBegin(GL_POINTS)
      glColor3f(0, 0, 255)
      glVertex3f(*closest_point)
      glEnd()
      glPointSize(5)

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

  def interpolate_color(self, c1, c2, alpha):
    return [
      c1[0] * alpha + c2[0] * (1.0 - alpha),
      c1[1] * alpha + c2[1] * (1.0 - alpha),
      c1[2] * alpha + c2[2] * (1.0 - alpha),
    ]
