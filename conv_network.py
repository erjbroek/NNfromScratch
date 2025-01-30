import numpy as np

class Convolution:
  # currently, code only supports relu activation
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
    # the kernel moves over the "picture". it starts at the top left corner and moves to the right
    # kernel can be seen as a small matrix of weights
    for k in range(self.num_kernels):
      for i in range(output_height):
        for j in range(output_width):
            # it looks at the surrounding values (based on kernel size). if the kernel size is 3 by 3, it looks at the direct surrounding values
            # if the kernel size is 5, it looks at the 2 surrounding values
            # then it multiplies the values of the pixels by the values in the kernel, and saves it to a new "picture" where the index is the middle index from the kernel
            # and sums them up to get a single value which is placed in the output feature map
          kernel_region = x_padded[i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size, :]
          output[i, j, k] = np.sum(kernel_region * self.kernels[k]) + self.bias[k]
    
    output = np.maximum(0, output)
    return output

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

       