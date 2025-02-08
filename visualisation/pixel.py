class Pixel:
    def __init__(self, canvas, pos_x, pos_y, width, height, color):
        self.canvas = canvas
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.height = height
        self.original_width = width
        self.original_height = height
        self.rect = canvas.create_rectangle(
            pos_x, pos_y, pos_x + width, pos_y + height, fill=color, outline=""
        )
        self.moved = False

    # transforms the pixels to show that each pixel is an input value for the neural network
    # after the pixels are in a long list, it is more clear that each pixel is a variable
    def transform(self, target_x, target_y, duration_ms, shrink_factor=1.0, steps=50, start_delay_ms=1000):
        self.target_x = target_x
        self.target_y = target_y
        self.start_x = self.pos_x
        self.start_y = self.pos_y
        self.steps = steps
        self.step_delay = duration_ms // steps
        
        final_width = self.original_width * shrink_factor
        final_height = self.original_height * shrink_factor
        self.start_width = self.width
        self.start_height = self.height
        self.dw = self.start_width - final_width
        self.dh = self.start_height - final_height

        self.current_step = 0

        self.canvas.after(start_delay_ms, self.animate)


    
    def animate(self):
        if self.current_step < self.steps:
            progress = self.current_step / self.steps
            eased_progress = progress * progress * (3 - 2 * progress)

            self.pos_x = self.start_x + (self.target_x - self.start_x) * eased_progress
            self.pos_y = self.start_y + (self.target_y - self.start_y) * eased_progress
            self.width = self.start_width - self.dw * eased_progress
            self.height = self.start_height - self.dh * eased_progress

            self.canvas.coords(
                self.rect,
                self.pos_x, self.pos_y,
                self.pos_x + self.width, self.pos_y + self.height
            )

            self.current_step += 1
            self.canvas.after(self.step_delay, self.animate)
        else:
            self.canvas.coords(
                self.rect,
                self.target_x, self.target_y,
                self.target_x + self.width, self.target_y + self.height
            )