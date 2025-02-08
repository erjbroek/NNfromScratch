class Node:
  def __init__(self, canvas, pos_x, pos_y, radius, value, delay, is_output=False):
    self.start_color = (41, 41, 41)
    self.hex_color = "#%02x%02x%02x" % self.start_color
    self.canvas = canvas
    self.pos_x = pos_x
    self.pos_y = pos_y
    self.value = value
    self.delay = delay
    self.next = None
    self.is_output = is_output
    self.oval = None
    self.end_color = (int(value * 255), int(value * 255), int(value * 255))
    self.end_color_outline = (70, 70, 70)
    self.canvas.after(self.delay, self.create_oval, pos_x, pos_y, radius)

  def create_oval(self, pos_x, pos_y, radius):
    self.oval = self.canvas.create_oval(
      pos_x, pos_y, pos_x + radius, pos_y + radius, fill=self.hex_color, outline=self.hex_color
    )
    self.fade_in(0)

  def fade_in(self, step):
    if step <= 100:
      fill_color = self.start_color[0] + (self.end_color[0] - self.start_color[0]) * step // 100
      outline_color = self.start_color[0] + (self.end_color_outline[0] - self.start_color[0]) * step // 100
      self.hex_color = "#%02x%02x%02x" % (fill_color, fill_color, fill_color)
      self.outline_color = "#%02x%02x%02x" % (outline_color, outline_color, outline_color)
      self.canvas.itemconfig(self.oval, fill=self.hex_color, outline=self.outline_color, width=1)
      self.canvas.after(self.delay // 100, self.fade_in, step + 1)

  def change_color(self, value):
    self.start_color = (0, 0, 0)
    self.end_color_outline = (41, 41, 41)
    self.end_color = (int(value * 255), int(value * 255), int(value * 255))
    self.fade_in(0)