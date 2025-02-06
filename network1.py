from sklearn.model_selection import train_test_split
import numpy as np
from hidden_layer import hidden_layer
from output_layer import output_layer
class NeuralNetwork1:
  def __init__(self, x_train, y_train, x_test, y_test, input_size, output_size, amount_hidden_layers, amount_nodes):
    self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
    self.input_size = input_size
    self.output_size = output_size
    print(f"Input size: {self.input_size}, Output size: {self.output_size}")
    self.network = []

    for i in range(amount_hidden_layers):
      if i == 0:
        # input layer, nodes equal to input size * amount of nodes in first hidden layer
        self.network.append(hidden_layer(self.input_size, amount_nodes[i], 0.01, 0.9, 0.999))
      else:
        self.network.append(hidden_layer(amount_nodes[i - 1], amount_nodes[i], 0.01, 0.9, 0.999))

    self.network.append(output_layer(self.network[-1].output_size, self.output_size, 0.01, 0.9, 0.999))

  def train(self, epochs, batch_size):
    n_samples = self.x_train.shape[0]
    loss = []
    accuracy = []

    for epoch in range(epochs):
      indices = np.arange(n_samples)
      np.random.shuffle(indices)
      x = self.x_train[indices]
      y = self.y_train[indices]

      for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_x = x[start:end]
        batch_y = y[start:end]

        for layer in self.network:
          batch_x = layer.forward(batch_x)
        y_hat = batch_x
        
        for layer in reversed(self.network):
          #output layer, where initial gradient is calculated
          if isinstance(layer, output_layer):
            previous_hidden_activation = self.network[-2].activation
            gradient = layer.backward(previous_hidden_activation, y_hat, batch_y)

          # hidden layer, which uses the gradient from the next layer
          # elif self.network.index(layer) > 0:
          elif self.network.index(layer) > 0:
            previous_hidden_activation = self.network[self.network.index(layer) - 1].activation
            previous_hidden_weights = self.network[self.network.index(layer) + 1].weights
            gradient = layer.backward(previous_hidden_activation, previous_hidden_weights.T, gradient)
          else:
            previous_hidden_activation = batch_x
            previous_hidden_weights = self.network[self.network.index(layer) + 1].weights
            gradient = layer.backward(previous_hidden_activation, previous_hidden_weights.T, gradient)



        x = self.x_test
        for layer in self.network:
          x = layer.forward(x)
        y_hat = x

        predictions = np.argmax(y_hat, axis=1)
        targets = np.argmax(self.y_test, axis=1)
        acc = np.mean(predictions == targets) * 100
        accuracy.append(acc)
        print(f'Epoch: {epoch+1}/{epochs}, Accuracy: {acc}')
      return loss, accuracy
      
