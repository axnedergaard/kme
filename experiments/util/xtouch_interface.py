import mido
import time
import threading

# TODO. This sometimes crashes. It is unclear if this is a mutex issue, but seems like it's not.

xtouch_interface = None
def get_xtouch_interface(parameters = {}):
  global xtouch_interface
  if xtouch_interface is None:
    xtouch_interface = XTouchInterface(parameters)
  elif parameters != {}:
    xtouch_interface.add_parameters(parameters)
  return xtouch_interface

class XTouchInterface:
  def __init__(self, parameters, presets=None):
    self.default_slider_low = -8192
    self.default_slider_high = 8191
    self.n_channels = 8
    self.n_lcd_rows = 2
    self.n_lcd_cols = 7
    port_name = 'X-Touch-Ext:X-Touch-Ext X-TOUCH_INT 20:0' # TODO

    available_ports = mido.get_output_names()
    assert port_name in available_ports
    self.input = mido.open_input(port_name)
    self.output = mido.open_output(port_name)
    
    self.lock = threading.Lock()

    self.clear()
    self.presets = presets
    self.preset_index = 0
    self.n_parameters = 0 
    self.slider_names = []
    self.slider_low = []
    self.slider_high = []
    self.slider_values = {}
    self.add_parameters(parameters)

    self.thread = threading.Thread(target=self.read)
    self.thread.start()

  def add_parameters(self, parameters):
    for parameter in parameters:
      self.add_parameter(parameter[0], parameter[1], parameter[2])

  def add_parameter(self, name, low, high):
    assert high >= low
    index = self.n_parameters
    self.slider_names.append(name)
    self.slider_low.append(low)
    self.slider_high.append(high)
    value = (low + high) // 2
    self.set_value(index, value, value_is_pitch=False)
    self.write(name, index, 0)
    self.n_parameters += 1

  def rescale(self, value, old_min, old_max, new_min, new_max):
    return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

  def set_value(self, index, value, value_is_pitch=True):
    if value_is_pitch:
      pitch = value
      value = self.rescale(value, self.default_slider_low, self.default_slider_high, self.slider_low[index], self.slider_high[index])
    else:
      pitch = int(self.rescale(value, self.slider_low[index], self.slider_high[index], self.default_slider_low, self.default_slider_high))
    name = self.slider_names[index]
    self.lock.acquire()
    self.slider_values[name] = value 
    self.lock.release()
    self.write('{:.2f}'.format(value), index, 1)
    # The slider annoying going back to the last position set with output.send. 
    msg = mido.Message('pitchwheel', channel=index, pitch=pitch)
    self.output.send(msg)

  def set_preset(self, forward):
    if forward:
      self.preset_index = (self.preset_index + 1) % len(self.presets)
    else:
      self.preset_index = (self.preset_index - 1) % len(self.presets)
    preset = self.presets[self.preset_index]
    for index, value in enumerate(preset):
      self.set_value(index, value, value_is_pitch=False)

  def get_values(self):
    self.lock.acquire()
    values = self.slider_values
    self.lock.release()
    return values 

  def read(self):
    print(self)
    for msg in self.input:
      if msg.type == 'pitchwheel':
        index = msg.channel
        if index >= self.n_parameters: 
          continue
        self.set_value(index, msg.pitch)
      elif msg.type == 'control_change':
        if self.presets is None:
          continue
        forward = msg.value in list(range(1,5))
        self.set_preset(forward)

  def write(self, text, index, row):
    assert index <= self.n_channels
    assert row <= self.n_lcd_rows
    assert len(text) <= self.n_lcd_cols

    # Create message bytes.
    ascii_text = ''
    if len(text) <= self.n_lcd_cols: 
      ascii_text += ' ' * (self.n_lcd_cols - len(text)) # Pad with spaces.
      ascii_text = [ord(c) for c in text]
    ascii_text = ascii_text[:self.n_lcd_cols] # Truncate if necessary.

    lcd = 7 * index + 8 * 7 * row # Character number on all LCDs

    header = [0xF0, 0x00, 0x00, 0x66, 0x15, 0x12] # Constants.
    end = [0xF7]
    msg = header + [lcd] + ascii_text + end 

    # Create MIDI message from bytes.
    message = mido.Message.from_bytes(bytes(msg))

    self.output.send(message)

  def clear(self):
    for lcd_index in range(self.n_channels):
      for lcd_row in range(self.n_lcd_rows):
        self.write(' ' * self.n_lcd_cols, lcd_index, lcd_row)
