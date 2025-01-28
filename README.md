# Neural Network from Scratch (NumPy-based)

This project is a fully connected neural network implemented from scratch using NumPy. It currently supports batch gradient descent (BGD) and mini-batch gradient descent (MBGD) for training. Since this is a very basic implementation of a neural network, it won't be as accurate as more advanced models line cnn's.

## User Interface

Other than a neural network, a simple ui has been created with TKinter.

- **Digit Drawing Interface**: A simple user interface has been created using **Tkinter**. This allows users to draw their own digits on a canvas.
- **Digit Conversion**: The drawn digits are then converted into a 28x28 pixel array, which can be fed into the neural network.
- **Prediction of digits**: The neural network will give the probilities for each digits
- **Animation of forward propogation**: to give a clearer overview of how the predictions are made, an animation has been made of how it (roughly) works

<p align="center">
  <img src="https://github.com/user-attachments/assets/213075eb-8b9e-44c7-8cd3-2085500f661f" width="500">
  <img src="https://github.com/user-attachments/assets/8a88a6c6-b8e6-4cda-afd7-f3792fddfc7a" width="500">
</p>



## Features

- **Weight Initialization**: **He Initialization** is used for weight initialization to prevent issues like vanishing gradients
- **Hidden Layer Activation**: **ReLU** (Rectified Linear Unit) is used as the activation function for hidden layers, allowing the model to learn non-linear relationships.
- **Output Layer Activation**: **Softmax** is used for the output layer to calculate class probabilities, ideal for multi-class classification tasks.
- **Loss Function**: **Cross-Entropy Loss** is employed to quantify the difference between the true labels and predicted probabilities.
- **Mini batch gradient descend**: The network uses **Backpropagation** using both Mini batch gradient descent or batch gradient descent to optimize weights based on the gradients computed from the loss function.
- **Optimizer**: Uses the **Adam Optimizer**, which combines the benefits of AdaGrad and RMSProp for faster and better training.



## Performance

| Dataset       | Accuracy | hidden layer | learning rate |
|--------------|----------| -------- | ------- |
| [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) | 99.83%    | 1 hidden layer, 7 nodes | 0.0001 | 
| [MNIST](https://en.wikipedia.org/wiki/MNIST_database) | 98.44%    | 1 hidden layer, 392 nodes | 0.01 |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | 90.24%    | 1 hidden layer, 392 nodes | 0.001 |
| More datasets coming soon


## Future Plans

- Test on additional datasets
- Add support for multiple hidden layers

