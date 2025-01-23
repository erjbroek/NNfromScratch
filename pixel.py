class Pixel:
  def __init__(self, pos_x, pos_y, width, height, val):
    self.pos_x = pos_x
    self.pos_y = pos_y
    self.width = width
    self.height = height
    self.val = val

  def render(self, canvas, color):
    canvas.create_rectangle(self.pos_x, self.pos_y, self.pos_x + self.width, self.pos_y + self.height, fill=color)
