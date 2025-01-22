# Neural Network from Scratch (NumPy-based)

This project is a fully connected neural network implemented from scratch using NumPy. It currently supports batch gradient descent (BGD) and mini-batch gradient descent (MBGD) for training. Since this is a very basic implementation of a neural network, it won't be as accurate as more advanced models.

## Features

- **Weight Initialization**: **He Initialization** is used for weight initialization to prevent issues like vanishing gradients
- **Hidden Layer Activation**: **ReLU** (Rectified Linear Unit) is used as the activation function for hidden layers, allowing the model to learn non-linear relationships.
- **Output Layer Activation**: **Softmax** is used for the output layer to calculate class probabilities, ideal for multi-class classification tasks.
- **Loss Function**: **Cross-Entropy Loss** is employed to quantify the difference between the true labels and predicted probabilities.
- **Mini batch gradient descend**: The network uses **Backpropagation** using both Mini batch gradient descent and batch gradient descent to optimize weights based on the gradients computed from the loss function.


## Performance

| Dataset       | Accuracy | hidden layer | learning rate |
|--------------|----------| -------- | ------- |
| [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) | 97.83%    | 1 hidden layer, 7 nodes | 0.0001 | 
| [MNIST](https://en.wikipedia.org/wiki/MNIST_database) | 98.47%    | 1 hidden layer, 392 nodes | 0.01 |
| More datasets coming soon


## Future Plans

- Test on additional datasets
- Add UI to draw digits for network to predict
- Experimenting with optimisers
- Add support for multiple hidden layers

