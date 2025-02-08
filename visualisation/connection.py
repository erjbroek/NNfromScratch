class Connection:
  def __init__(self, canvas, x0, x1, y0, y1, weight, threshold, delay=10, start_delay=0):
    self.canvas = canvas
    self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
    self.weight = abs(weight)
    self.delay = delay
    
    weight_clamped = max(threshold, min(self.weight, 1.2 * threshold))
    ratio = (weight_clamped - threshold) / (1.2 * threshold - threshold)

    base_color = 0x29
    max_color = 0xFF
    interpolated_color = int(base_color + (max_color - base_color) * ratio)
    self.start_color = (base_color, base_color, base_color)
    self.end_color = (interpolated_color, interpolated_color, interpolated_color)
    self.hex_color = "#%02x%02x%02x" % self.start_color

    self.canvas.after(start_delay, self.fade_in, 0)

  def fade_in(self, step):
    if step <= 15:
      self.line = self.canvas.create_line(self.x0, self.y0, self.x1, self.y1, fill=self.hex_color, width=self.weight)
      fill_color = self.start_color[0] + (self.end_color[0] - self.start_color[0]) * step // 20  # Adjust for 20 steps
      self.hex_color = "#%02x%02x%02x" % (fill_color, fill_color, fill_color)

      self.canvas.itemconfig(self.line, fill=self.hex_color)
      self.canvas.after(self.delay // 20, self.fade_in, step + 1)
