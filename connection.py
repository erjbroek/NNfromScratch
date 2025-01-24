class Connection:
    def __init__(self, canvas, x0, x1, y0, y1, weight, threshold):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.weight = abs(weight)
        self.canvas = canvas

        weight_clamped = max(threshold, min(self.weight, 1.3 * threshold))
        ratio = (weight_clamped - threshold) / (1.3 * threshold - threshold)

        base_color = 0x29
        max_color = 0xFF
        interpolated_color = int(base_color + (max_color - base_color) * ratio)

        color_hex = "#%02x%02x%02x" % (interpolated_color, interpolated_color, interpolated_color)
        
        self.canvas.create_line(self.x0, self.y0, self.x1, self.y1, fill=color_hex, width=self.weight)
