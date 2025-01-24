import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from network import NeuralNetwork
import pandas as pd
from pixel import Pixel
from node import Node
from connection import Connection
from PIL import Image, ImageTk

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
epochs = 0
learning_rate = 0.01

loss, accuracy = network.train_mbgd(mnist_train_x, mnist_train_y, epochs, learning_rate, 64)
print(accuracy)

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        self.root.configure(bg="#292929")

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
        predictions = predictions[0]
        predictions = np.array([round(p * 100, 0) for p in predictions])
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
    
    # this is a very long method (which i want to split up in the future)
    # this gives a visual of how the forward propogation of the neural network works
    # it splits the image into the pixels, which are 'fed' into the neural network

    def animate_prediction(self):
        if not self.empty:
            animation_canvas = tk.Canvas(self.root, width=self.root.winfo_width(), height=self.root.winfo_height(), bg="#292929", highlightthickness=0)
            animation_canvas.place(x=0, y=0)

            # this number makes sure the image will be the same size as the original drawing canvas
            # to make the transition smoother from canvas to animation
            size = 13.5714
            pixels = []
            for i in range(28 * 28):
                row = i // 28
                col = i % 28
                pixel_color = f"#{int(self.img_flattened[0][i] * 255):02x}{int(self.img_flattened[0][i] * 255):02x}{int(self.img_flattened[0][i] * 255):02x}"
                pixel = Pixel(animation_canvas, 10 + col * size, 10 + row * size, size, size, pixel_color)
                pixels.append(pixel)

                x = 50
                y = min(40 + (i // 25) * 10, 350)
                if 180 <= y <= 230:
                    y += 50
                pixel.transform(x, y, 2000, 0.5, 50, 0)
            
            animation_canvas.after(2000, lambda: animation_canvas.create_text(54, 183, text=".", font=("Helvetica", 16), fill="white"))
            animation_canvas.after(2000, lambda: animation_canvas.create_text(54, 194, text=".", font=("Helvetica", 16), fill="white"))
            animation_canvas.after(2000, lambda: animation_canvas.create_text(54, 205, text=".", font=("Helvetica", 16), fill="white"))
            
            # creates and renders the nodes for each layer
            input_layer = []
            hidden_layer = []
            output_layer = []
            x = 160
            for i in range(27):
                y = i * 13 + 20
                node = Node(animation_canvas, x, y, 11, 0, 2000)
                input_layer.append(node)
            
            for i in range(7):
                y = i * 20 + 40
                node = Node(animation_canvas, x + 150, y, 13, 0, 2000)
                hidden_layer.append(node)

            # the dots to show there are more nodes than possible to fit on the screen
            animation_canvas.after(2000, lambda: animation_canvas.create_text(x + 155, 183, text=".", font=("Helvetica", 16), fill="white"))
            animation_canvas.after(2000, lambda: animation_canvas.create_text(x + 155, 194, text=".", font=("Helvetica", 16), fill="white"))
            animation_canvas.after(2000, lambda: animation_canvas.create_text(x + 155, 205, text=".", font=("Helvetica", 16), fill="white"))

            for i in range(8, 15):
                y = i * 20 + 60
                node = Node(animation_canvas, x + 150, y, 13, 0, 2000)
                hidden_layer.append(node)

            for i in range(10):
                y = i * 22 + 70
                node = Node(animation_canvas, x + 300, y, 15, 0, 2000, is_output=True)
                output_layer.append(node)

            # creates connections
            # since there are too many connections (if i rendered them all, there would be about 550)
            # i only render the ones with a high enough weight
            # this only shows the weights with having more than 0 epochs, since they are initialised
            # at a very low number
            threshold = 0.09
            for i in range(len(input_layer) * len(hidden_layer)):
                row = i // len(hidden_layer)
                col = i % len(hidden_layer)
                connection_weight = network.W_IH[row, col]
                if abs(connection_weight) > threshold:
                    x0 = input_layer[row].pos_x + 7
                    y0 = input_layer[row].pos_y + 7
                    x1 = hidden_layer[col].pos_x + 7
                    y1 = hidden_layer[col].pos_y + 7
                    Connection(animation_canvas, x0, x1, y0, y1, connection_weight, threshold, 1000, 3000)
            
            threshold = 0.2
            for i in range(len(hidden_layer) * len(output_layer)):
                row = i // len(output_layer)
                col = i % len(output_layer)
                connection_weight = network.W_HO[row, col]
                if abs(connection_weight) > threshold:
                    x0 = hidden_layer[row].pos_x + 7
                    y0 = hidden_layer[row].pos_y + 7
                    x1 = output_layer[col].pos_x + 7
                    y1 = output_layer[col].pos_y + 7
                    Connection(animation_canvas, x0, x1, y0, y1, connection_weight, threshold, 1000, 3000)

            # this moves the pixels again after the connections have been created
            # for some reason i haven't figured out yet, the transform method also gets called
            # the first time the pixels get moved (even though it should have a delay of 99999)
            # with this reason, a seperate method has been made, which somehow fixes it
            animation_canvas.after(7000, lambda: self.move_pixels_to_new_location(pixels))
               
    def move_pixels_to_new_location(self, pixels):
        for i, pixel in enumerate(pixels):
            new_x = 160
            new_y = min((i // 27 * 13 + 20), 370)
            pixel.transform(new_x, new_y, 2000, 0.5, 50, 999999)

root = tk.Tk()
root.geometry("800x400")
app = DigitApp(root)
root.mainloop()
