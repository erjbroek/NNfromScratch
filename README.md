# Neural Network from Scratch (NumPy-based)

This project is a fully connected neural network implemented from scratch using NumPy. It currently supports batch gradient descent (BGD) and mini-batch gradient descent (MBGD) for training. The network has been tested on the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), achieving an accuracy of 98% after training. Future plans include testing on additional datasets.

Since this is a very basic implementation of a neural network, it won't be as accurate as more advanced models.

## Features

- **Weight Initialization**: **He Initialization** is used for weight initialization to prevent issues like vanishing gradients
- **Hidden Layer Activation**: **ReLU** (Rectified Linear Unit) is used as the activation function for hidden layers, allowing the model to learn non-linear relationships.
- **Output Layer Activation**: **Softmax** is used for the output layer to calculate class probabilities, ideal for multi-class classification tasks.
- **Loss Function**: **Cross-Entropy Loss** is employed to quantify the difference between the true labels and predicted probabilities.
- **Optimization Algorithm**: The network uses **Backpropagation with Gradient Descent** to optimize weights based on the gradients computed from the loss function.
  - Supports **Batch Gradient Descent** and **Mini-Batch Gradient Descent** as optimization techniques.

## Performance

| Dataset       | Accuracy |
|--------------|----------|
| [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) | 98%      |
| More datasets coming soon | - |


## Future Plans

- Test on additional datasets (like [MNIST](https://en.wikipedia.org/wiki/MNIST_database))
- Experimenting with optimisers
- Add support for multiple hidden layers

