import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from network import NeuralNetwork
import pandas as pd
from PIL import Image, ImageTk
from animation import Animations

mnist_train = pd.read_csv('./mnist/mnist_train.csv', header=None)
mnist_test = pd.read_csv('./mnist/mnist_test.csv', header=None)

mnist_train_x = mnist_train.iloc[:, 1:].values / 255
mnist_train_y = np.eye(10)[mnist_train.iloc[:, 0].values]
mnist_test_x = mnist_test.iloc[:, 1:].values / 255
mnist_test_y = np.eye(10)[mnist_test.iloc[:, 0].values]

input_size = len(mnist_train_x[0])
output_size = len(np.unique(mnist_train.iloc[:, 0].values))
hidden_size = input_size // 2

network = NeuralNetwork(input_size, hidden_size, output_size)
epochs = 1
learning_rate = 0.01

loss, accuracy = network.train_mbgd(mnist_train_x, mnist_train_y, epochs, learning_rate, 64)
print(accuracy)

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        self.root.configure(bg="#292929")

        self.animations = Animations(network)

        self.empty = True
        self.canvas = tk.Canvas(root, width=380, height=380, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.canvas.place(x=10, y=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.button = tk.Button(root, text="Clear", command=self.clear, bg="#292929", fg="white", font=("Helvetica", 12), width=10, highlightbackground="#292929", highlightthickness=1)
        self.button.pack()
        self.button.place(x=400, y=10)

        self.button = tk.Button(root, text="Animate", command=self.animate_prediction, bg="#292929", fg="white", font=("Helvetica", 12), width=10, highlightbackground="#292929", highlightthickness=1)
        self.button.pack()
        self.button.place(x=400, y=250)

        self.predictions = None
        
        self.image = Image.new("L", (380, 380), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.img_flattened = np.zeros((1, 784))
        self.display_image(self.img_flattened)

    def paint(self, event):
        self.empty = False
        x1, y1 = (event.x - 18), (event.y - 16)
        x2, y2 = (event.x + 16), (event.y + 16)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)
        self.process_output()
    
    def clear(self):
        self.empty = True
        self.canvas.delete("all")
        self.image = Image.new("L", (380, 380), 0)
        self.draw = ImageDraw.Draw(self.image)
        
    def process_output(self):
        img = self.image.resize((28, 28))
        img = np.array(img) / 255
        
        self.img_flattened = img.reshape(1, -1)
        predictions = network.predict(self.img_flattened)
        self.display_image(img)
        self.display_output(predictions)

    def display_output(self, predictions):
        pos_x = 100
        output_canvas = tk.Canvas(self.root, width=140 + pos_x, height=380, bg="#292929", highlightthickness=0)
        output_canvas.place(x=600 - pos_x, y=10)
        self.predictions = predictions[0]
        predictions = np.array([round(p * 100, 0) for p in self.predictions])
        for i, prediction in enumerate(predictions):
            if i == predictions.argmax():
                output_canvas.create_text(20 + pos_x, 20 + (i * 37), text=f"{i}", font=("Helvetica", 12, "bold"), fill="white")
            else:
                output_canvas.create_text(20 + pos_x, 20 + (i * 37), text=f"{i}", font=("Helvetica", 12), fill="white")
                
            color = int(41 + (prediction * 2.14))
            hex_color = f"#{color:02x}{color:02x}{color:02x}"
            output_canvas.create_oval(30 + pos_x, 6 + (i * 37), 60 + pos_x, 37 + (i * 37), fill=hex_color, outline='white')
            output_canvas.create_text(95 + pos_x, 20 + (i * 37), text=f"{prediction}%", font=("Helvetica", 12), fill="white")
            output_canvas.create_line(0, 340, pos_x, 20 + (i * 37), fill="white")

    def display_image(self, img):
        img_array = (np.array(img).reshape((28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img = img.resize((110, 110), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(img)

        image_canvas = tk.Canvas(self.root, width=100, height=100, highlightthickness=0, bd=0)
        image_canvas.place(x=400, y=290)
        image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def get_image(self):
        return self.img_flattened

    def animate_prediction(self):
        if not self.empty:
            animation_canvas = tk.Canvas(self.root, width=self.root.winfo_width(), height=self.root.winfo_height(), bg="#292929", highlightthickness=0)
            animation_canvas.place(x=0, y=0)

            pixels = self.animations.move_pixels_left(animation_canvas, self.img_flattened)
            
            input_layer = self.animations.spawn_inputnodes(animation_canvas, 160)
            hidden_layer = self.animations.spawn_hiddennodes(animation_canvas, 260)
            output_layer = self.animations.spawn_outputnodes(animation_canvas, 360)

            connections_i_h, connections_h_o = self.animations.spawn_connections(animation_canvas, input_layer, hidden_layer, output_layer)

            animation_canvas.after(9000, lambda: self.move_pixels_to_new_location(pixels))
            animation_canvas.after(14000, lambda c=connections_i_h, p=pixels: self.animations.animate_feedforward(c, p, 410, 0.5))
            animation_canvas.after(19000, lambda c=connections_h_o, p=pixels: self.animations.animate_feedforward(c, p, 650, 0.1))

            animation_canvas.after(20000, lambda: self.animations.color_output_values(self.predictions, output_layer))
            animation_canvas.after(21000, lambda: self.render_remove_button(animation_canvas))

    def render_remove_button(self, animation_canvas):
        self.remove_button = tk.Button(self.root, text="Return", command=lambda: self.remove_animation_canvas(animation_canvas), bg="#292929", fg="white", font=("Helvetica", 12), width=15, highlightbackground="#292929", highlightthickness=1)
        self.remove_button.place(x=10, y=10)

    def remove_animation_canvas(self, animation_canvas):
        animation_canvas.destroy()
        self.remove_button.destroy()
        self.canvas.lift("all")

    def move_pixels_to_new_location(self, pixels):
        for i, pixel in enumerate(pixels):
            new_x = 167
            new_y = min((i // 27 * 13 + 27), 370)
            pixel.transform(new_x, new_y, 2000, 0.5, 50, 999)


root = tk.Tk()
root.geometry("800x400")
app = DigitApp(root)
root.mainloop()
