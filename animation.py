from collections import defaultdict
from connection import Connection
from node import Node
import random
from pixel import Pixel
import tkinter as tk


class Animations:
    def __init__(self, network):
        self.network = network
        pass

    def move_pixels_left(self, canvas, image,):

        # this number makes sure the image will be the same size as the original drawing canvas
        # to make the transition smoother from canvas to animation
        size = 13.5714
        pixels = []
        for i in range(784):
            row = i // 28
            col = i % 28
            pixel_color = f"#{int(image[0][i] * 255):02x}{int(image[0][i] * 255):02x}{int(image[0][i] * 255):02x}"
            pixel = Pixel(canvas, 10 + col * size, 10 + row * size, size, size, pixel_color)
            pixels.append(pixel)
            canvas.after(2000, lambda: canvas.create_text(54, 183, text=".", font=("Helvetica", 16), fill="white"))
            canvas.after(2000, lambda: canvas.create_text(54, 194, text=".", font=("Helvetica", 16), fill="white"))
            canvas.after(2000, lambda: canvas.create_text(54, 205, text=".", font=("Helvetica", 16), fill="white"))
            x = 50
            y = min(40 + (i // 25) * 10, 350)
            if 180 <= y <= 230:
                y += 50
            pixel.transform(x, y, 2000, 0.5, 50, 0)
        return pixels

    def spawn_inputnodes(self, canvas, x):
        layer = []
        for i in range(27):
            y = i * 13 + 20
            node = Node(canvas, x, y, 11, 0, 2000)
            layer.append(node)
        return layer

    def spawn_hiddennodes(self, canvas, x):
        layer = []       
        for i in range(7):
            y = i * 20 + 40
            node = Node(canvas, x + 150, y, 13, 0, 2000)
            layer.append(node)

        # the dots to show there are more nodes than possible to fit on the screen
        canvas.after(2000, lambda: canvas.create_text(x + 155, 183, text=".", font=("Helvetica", 16), fill="white"))
        canvas.after(2000, lambda: canvas.create_text(x + 155, 194, text=".", font=("Helvetica", 16), fill="white"))
        canvas.after(2000, lambda: canvas.create_text(x + 155, 205, text=".", font=("Helvetica", 16), fill="white"))

        for i in range(8, 15):
            y = i * 20 + 60
            node = Node(canvas, x + 150, y, 13, 0, 2000)
            layer.append(node)
        return layer


    def spawn_outputnodes(self, canvas, x):
        layer = []
        for i in range(10):
            y = i * 22 + 70
            node = Node(canvas, x + 300, y, 15, 0, 2000, is_output=True)
            layer.append(node)
        return layer

    def spawn_connections(self, animation_canvas, input_layer, hidden_layer, output_layer):
        connections_i_h = []
        threshold = 0.075
        for i in range(len(input_layer) * len(hidden_layer)):
            row = i // len(hidden_layer)
            col = i % len(hidden_layer)
            connection_weight = self.network.W_IH[row, col]
            if abs(connection_weight) > threshold:
                x0 = input_layer[row].pos_x + 7
                y0 = input_layer[row].pos_y + 7
                x1 = hidden_layer[col].pos_x + 7
                y1 = hidden_layer[col].pos_y + 7
                connection = Connection(animation_canvas, x0, x1, y0, y1, connection_weight, threshold, 1000, 3000)
                connections_i_h.append(connection)
            
        connections_h_o = []
        threshold = 0.18
        for i in range(len(hidden_layer) * len(output_layer)):
            row = i // len(output_layer)
            col = i % len(output_layer)
            connection_weight = self.network.W_HO[row, col]
            if abs(connection_weight) > threshold:
                x0 = hidden_layer[row].pos_x + 7
                y0 = hidden_layer[row].pos_y + 7
                x1 = output_layer[col].pos_x + 7
                y1 = output_layer[col].pos_y + 7
                connection = Connection(animation_canvas, x0, x1, y0, y1, connection_weight, threshold, 1000, 3000)
                connections_h_o.append(connection)

        return connections_i_h, connections_h_o

    def animate_feedforward(self, connections, pixels, x, shrink_factor=0.5):
        connection_map = defaultdict(list)
        for connection in connections:
            connection_map[connection.y0].append(connection.y1)

        for pixel in pixels:
            current_pos = round(pixel.pos_y)
            if current_pos in connection_map:
                possible_destinations = connection_map[current_pos]
                new_y = random.choice(possible_destinations)
                pixel.transform(x, new_y, 2000, shrink_factor, 50)

    def color_output_values(self, predictions, output_layer):
        for i, node in enumerate(output_layer):
            node.change_color(predictions[i])
            if i == predictions.argmax():
                node.canvas.create_text(node.pos_x + 20, node.pos_y + 7, text=f"{i}: {round(predictions[i] * 100, 2)}% certain", font=("Helvetica", 12), fill="white", anchor=tk.W)