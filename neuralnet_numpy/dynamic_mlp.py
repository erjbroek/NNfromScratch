from sklearn.model_selection import train_test_split
import numpy as np
from models.layers.hidden_layer import hidden_layer
from models.layers.output_layer import output_layer
from augment_mnist import augment

class dynamic_mlp:
  def __init__(self, x_train, y_train, x_test, y_test, input_size, output_size, amount_hidden_layers, amount_nodes, learning_rate, should_augment):
    self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
    self.input_size = input_size
    self.output_size = output_size
    print(f"Input size: {self.input_size}, Output size: {self.output_size}")
    self.network = []
    self.should_augment = should_augment
    for i in range(amount_hidden_layers):
      if i == 0:
        # input layer, amount of nodes equal to the amount of pixels
        self.network.append(hidden_layer(self.input_size, amount_nodes[i], learning_rate, 0.9, 0.999))
      else:
        # hidden layer
        self.network.append(hidden_layer(amount_nodes[i - 1], amount_nodes[i], learning_rate, 0.9, 0.999))

    # output layer, amount of nodes equal to amount of classes.
    self.network.append(output_layer(self.network[-1].output_size, self.output_size, learning_rate, 0.9, 0.999))

  def train(self, epochs, batch_size):
    n_samples = self.x_train.shape[0]
    loss = []
    accuracy = []
    
    for epoch in range(epochs):
      indices = np.arange(n_samples)
      np.random.shuffle(indices)

      if self.should_augment:
        # x is redefined a few times here, since i plan to add visualisations in the future where i want to use the non normalised data for example
        x = self.x_train[indices]
        y = self.y_train[indices]
        x = augment(x, y)
        x = x / 255
      else:
        x = self.x_train[indices] / 255
        y = self.y_train[indices]


      for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_x = x[start:end]
        batch_y = y[start:end]

        for layer in self.network:
          batch_x = layer.forward(batch_x)
        y_hat = batch_x
        
        for layer in reversed(self.network):
          # output layer, where initial gradient is calculated
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
            previous_hidden_activation = x[start:end]
            previous_hidden_weights = self.network[self.network.index(layer) + 1].weights
            gradient = layer.backward(previous_hidden_activation, previous_hidden_weights.T, gradient)

      test_x = self.x_test
      for layer in self.network:
        test_x = layer.forward(test_x)
      y_hat = test_x

      predictions = np.argmax(y_hat, axis=1)
      targets = np.argmax(self.y_test, axis=1)
      acc = np.mean(predictions == targets) * 100
      accuracy.append(acc)
      print(f'Epoch: {epoch+1}/{epochs}, Accuracy: {acc}')
    print(accuracy[-1])
    return loss, accuracy
      
