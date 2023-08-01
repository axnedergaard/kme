from OpenGL.GL import *
import glfw

from . import xtouch_interface


# TODO. Interface sometimes crashes for unknown reason.
# TODO. We want the y rotation to not depend on the x rotation.

class Visualizer:
  def __init__(self):
    self.data = {}

    if not glfw.init():
      print("Failed to initialize OpenGL context")
    self.window = glfw.create_window(1000, 1000, "Visualizer", None, None)
    glfw.make_context_current(self.window)
    glPointSize(5)
    self.x_angle = 0
    self.y_angle = 0
    
    # Create interface.
    parameters = [
      ['x_angle', -180, 180],
      ['y_angle', -180, 180],
      ['scale', 0.5, 2],
    ]
    self.interface = xtouch_interface.XTouchInterface(parameters)

  def add(self, data):
    name = data['name']
    if name not in self.data: # Create new entry.
      data = data.copy()
      # Pad if needed.
      point_dim = data['points'].shape[1]
      if point_dim < 3:
        data['points'] = np.pad(data['points'], ((0,0), (0, 3-point_dim)), 'constant', constant_values=0)
      data['points'] = data['points'].tolist()
      del data['name']
      self.data.update({name: data})
    else: # Update existing entry.
      self.data[name]['points'] += data['points'].tolist()
      if data['color'] is not None:
        self.data[name]['color'] = data['color']

  def remove(self, name):
    self.data.pop(name, None)

  def render(self):
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
    # TODO. There is a bug here.
    glRotatef(self.y_angle, 0, 1, 0) 
    glRotatef(self.x_angle, 1, 0, 0) 
    glScalef(self.scale, self.scale, self.scale)
    # Draw.
    glBegin(GL_POINTS)
    for data in self.data.values():
      for point in data['points']:
        glColor3f(*data['color'])
        glVertex3f(*point) # Warning, probably will throw an error since using numpy array.
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
