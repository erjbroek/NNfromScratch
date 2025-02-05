import numpy as np

class output_layer:
  def __init__(self, input_size, output_size, learning_rate=0.01, beta1=0.9, beta2=0.999):
    self.input_size = input_size
    self.output_size = output_size
    self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
    self.bias = np.zeros(output_size)
    
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.timestep = 0

    self.momentum_weights = np.zeros_like(self.weights)
    self.adaptive_lr_weights = np.zeros_like(self.weights)
    self.momentum_bias = np.zeros_like(self.bias)
    self.adaptive_lr_bias = np.zeros_like(self.bias)

  def softmax(self, x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
  
  def cost(self, y_hat, y, deriv=False):
    if deriv:
      return y_hat - y
    y_hat = np.clip(y_hat, 1e-15, 1. - 1e-15)
    return -np.mean(np.sum(y * np.log(y_hat), axis=-1))

  def feedforward(self, x):
    self.z = np.dot(x, self.weights) + self.bias
    self.activation = self.softmax(self.z)
    return self.activation
  
  def backward(self, hidden_output, y_hat, y):
    self.timestep += 1
    gradient_output = self.cost(y_hat, y, deriv=True)

    gradient_weights = np.dot(hidden_output.T, gradient_output)
    gradient_bias = np.sum(gradient_output, axis=0)

    self.momentum_weights = self.beta1 * self.momentum_weights + (1 - self.beta1) * gradient_weights
    self.adaptive_lr_weights = self.beta2 * self.adaptive_lr_weights + (1 - self.beta2) * (gradient_weights ** 2)
    self.momentum_bias = self.beta1 * self.momentum_bias + (1 - self.beta1) * gradient_bias
    self.adaptive_lr_bias = self.beta2 * self.adaptive_lr_bias + (1 - self.beta2) * (gradient_bias ** 2)

    corrected_momentum_weights = self.momentum_weights / (1 - self.beta1 ** self.timestep)
    corrected_adaptive_lr_weights = self.adaptive_lr_weights / (1 - self.beta2 ** self.timestep)
    corrected_momentum_bias = self.momentum_bias / (1 - self.beta1 ** self.timestep)
    corrected_adaptive_lr_bias = self.adaptive_lr_bias / (1 - self.beta2 ** self.timestep)

    self.weights -= self.learning_rate * corrected_momentum_weights / (np.sqrt(corrected_adaptive_lr_weights) + 1e-8)
    self.bias -= self.learning_rate * corrected_momentum_bias / (np.sqrt(corrected_adaptive_lr_bias) + 1e-8)  

    return gradient_output



