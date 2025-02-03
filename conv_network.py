import numpy as np
from scipy import signal

class Convolution:
  # currently, code only supports relu activation
  def __init__(self, num_kernels, kernal_size, stride, padding, activation='relu'):
    self.num_kernels = num_kernels
    self.kernel_size = kernal_size
    self.stride = stride
    self.padding = padding
    self.activation = activation
    self.input = None
    self.kernels = np.random.randn(num_kernels, kernal_size, kernal_size) * np.sqrt(2.0 / num_kernels)
    self.bias = np.zeros(num_kernels)

  def forward(self, x):
    batch_size = x.shape[0]
    x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
    self.input = x_padded
    InputHeight, InputWidth = x.shape[1], x.shape[2]
    output_height = (InputHeight - self.kernel_size + 2 * self.padding) // self.stride + 1
    output_width = (InputWidth - self.kernel_size + 2 * self.padding) // self.stride + 1
    output = np.zeros((batch_size, output_height, output_width, self.num_kernels))

    for k in range(self.num_kernels):
      for i in range(output_height):
        for j in range(output_width):
          kernel_region = x_padded[:, i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size, :]
          output[:, i, j, k] = np.sum(kernel_region * self.kernels[k], axis=(1, 2, 3)) + self.bias[k]

    output = np.maximum(0, output)
    return output
    
  
  def backpropagation(self, output_gradient, learning_rate):
    batch_size, _, input_height, input_width = self.input.shape
    _, num_kernels, output_height, output_width = output_gradient.shape

    input_gradient = np.zeros_like(self.input)
    kernel_gradient = np.zeros_like(self.kernels)
    bias_gradient = np.zeros_like(self.bias)

    for b in range(batch_size):
      for k in range(num_kernels):
        for i in range(output_height):
          for j in range(output_width):
            i_start, j_start = i * self.stride, j * self.stride
            region = self.input[b, :, i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size]
            kernel_gradient[k] += output_gradient[b, k, i, j] * region
            input_gradient[b, :, i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size] += (
              output_gradient[b, k, i, j] * self.kernels[k]
            )
        bias_gradient[k] += np.sum(output_gradient[b, k])

    kernel_gradient /= batch_size
    bias_gradient /= batch_size

    self.kernels -= learning_rate * kernel_gradient
    self.bias -= learning_rate * bias_gradient

    return input_gradient
  
  # yes, this uses scipy's signal, but the correlate and convolve functions are just much faster than my code
  def backpropagation1(self, output_gradient, learning_rate):
    batch_size = self.input.shape[0]
    kernel_gradient = np.zeros_like(self.kernels)
    for b in range(batch_size):
        for k in range(self.kernels.shape[0]):
            kernel_gradient[k] += signal.correlate(self.input[b], output_gradient[b, k], mode='valid')

    kernel_gradient /= batch_size
    bias_gradient = np.sum(output_gradient, axis=(0, 2, 3)) / batch_size

    input_gradient = np.zeros_like(self.input)
    for b in range(batch_size):
        for k in range(self.kernels.shape[0]):
            input_gradient[b] += signal.convolve(output_gradient[b, k], np.flip(self.kernels[k]), mode='full')

    self.kernels -= learning_rate * kernel_gradient
    self.bias -= learning_rate * bias_gradient

    return input_gradient


class Pooling:
  def __init__(self, pool_size, stride, pooling_type, padding):
    self.pool_size = pool_size
    self.stride = stride
    self.pooling_type = pooling_type
    self.padding = padding

  def forward(self, x):
    if self.padding > 0:
      x = np.pad(x, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

    InputHeight, InputWidth, ChannelCount = x.shape
    output_height = (InputHeight - self.pool_size) // self.stride + 1
    output_width = (InputWidth - self.pool_size) // self.stride + 1
    output = np.zeros((output_height, output_width, ChannelCount))

    # like the convolutional layer, this moves over the "image"
    # instead of multiplying the values with the weigts from the kernel matrix, it simply gets the average or max value
    # this is done to reduce the size of the image while still keeping the most important parts 
    # (results of this often look like the image has been blurred)
    for i in range(output_height):
      for j in range(output_width):
        region = x[i * self.stride : i * self.stride + self.pool_size, j * self.stride : j * self.stride + self.pool_size, :]

        for c in range(ChannelCount):
          if self.pooling_type == 'max':
            output[i, j, c] = np.max(region[:, :, c])
          elif self.pooling_type == 'average':
            output[i, j, c] = np.mean(region[:, :, c])
    output = np.maximum(0, output)
    return output

       