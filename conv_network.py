import numpy as np

class Convolution:
  def __init__(self, num_kernals, kernal_size, stride, padding):
    self.num_kernals = num_kernals
    self.kernal_size = kernal_size
    self.stride = stride
    self.padding = padding
    self.kernals = np.random.randn(num_kernals, kernal_size, kernal_size) * np.sqrt(2.0 / num_kernals)
    self.bias = np.zeros(num_kernals)


class Pooling:
  def __init__(self, pool_size, stride, pooling_type):
    self.pool_size = pool_size
    self.stride = stride
    self.pooling_type = pooling_type

       