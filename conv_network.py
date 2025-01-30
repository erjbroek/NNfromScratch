import numpy as np

class Convolution:
  # currently, it only supports images that are square
  def __init__(self, num_kernals, kernal_size, stride, padding, activation, input_shape=None):
    self.num_kernals = num_kernals
    self.kernal_size = kernal_size
    self.stride = stride
    self.padding = padding
    self.activation = activation
    self.input_shape = input_shape
    self.kernals = np.random.randn(num_kernals, kernal_size, kernal_size) * np.sqrt(2.0 / num_kernals)
    self.bias = np.zeros(num_kernals)

  def forward(self, x):
    x_padded = np.pad(x, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
    InputHeight, InputWidth, ChannelCount = x.shape
    output_height = (InputHeight - self.kernel_size + 2 * self.padding) // self.stride + 1
    output_width = (InputWidth - self.kernel_size + 2 * self.padding) // self.stride + 1
    output = np.zeros((output_height, output_width, self.num_kernels))

    # The actual convolution
    # the width and height are the input dimensions
    # the depth is the number of kernels
    for k in range(self.num_kernels):
      for i in range(output_height):
        for j in range(output_width):
          region = x_padded[i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size, :]
          output[i, j, k] = np.sum(region * self.kernels[k]) + self.bias[k]
                    
    return output

    


class Pooling:
  def __init__(self, pool_size, stride, pooling_type):
    self.pool_size = pool_size
    self.stride = stride
    self.pooling_type = pooling_type

       