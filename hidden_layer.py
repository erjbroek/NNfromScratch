import numpy as np

class hidden_layer:
  def __init__(self, input_size, output_size, learning_rate, beta1, beta2):
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

  def relu(self, x, is_deriv=False):
    if is_deriv:
      return (x > 0).astype(float)
    return np.maximum(0, x)

  def forward(self, x):
    print('hidden forward')
    print(f"x: {x.shape}, weights: {self.weights.shape}, bias: {self.bias.shape}")
    self.z = np.dot(x, self.weights) + self.bias
    self.activation = self.relu(self.z)
    return self.activation
  
  def backward(self, x, weights, error):
    print("backwards")
    self.timestep += 1
    print(f"error: {error.shape}, weights: {weights.shape}")
    error = np.dot(error, weights) * self.relu(self.z, is_deriv=True)

    weights_gradient = np.dot(x.T, error)
    bias_gradient = np.sum(error, axis=0)

    print(f"momentum_weights: {self.momentum_weights.shape}, weights_gradient: {weights_gradient.shape}")
    self.momentum_weights = self.beta1 * self.momentum_weights + (1 - self.beta1) * weights_gradient
    self.adaptive_lr_weights = self.beta2 * self.adaptive_lr_weights + (1 - self.beta2) * (weights_gradient ** 2)
    self.momentum_bias = self.beta1 * self.momentum_bias + (1 - self.beta1) * bias_gradient
    self.adaptive_lr_bias = self.beta2 * self.adaptive_lr_bias + (1 - self.beta2) * (bias_gradient ** 2)

    corrected_momentum_weights = self.momentum_weights / (1 - self.beta1 ** self.timestep)
    corrected_adaptive_lr_weights = self.adaptive_lr_weights / (1 - self.beta2 ** self.timestep)
    corrected_momentum_bias = self.momentum_bias / (1 - self.beta1 ** self.timestep)
    corrected_adaptive_lr_bias = self.adaptive_lr_bias / (1 - self.beta2 ** self.timestep)

    self.weights -= self.learning_rate * corrected_momentum_weights / (np.sqrt(corrected_adaptive_lr_weights) + 1e-8)
    self.bias -= self.learning_rate * corrected_momentum_bias / (np.sqrt(corrected_adaptive_lr_bias) + 1e-8)
    return error
  