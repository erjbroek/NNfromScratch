import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas
import matplotlib.pyplot as plt

def augment(X, y):
  X = X.values.reshape(-1, 28, 28, 1) / 255.0
  X_augmented = np.empty_like(X)
    
  for i in range(len(X)):
    # beta is for rotation or horizontal shearing
    beta = np.random.uniform(-7.5, 7.5) if y[i] in [1, 7] else np.random.uniform(-15.0, 15.0)

    # gamma is for horizontal and vertical scaling
    gamma = np.random.uniform(15, 20)
    scale_x = np.random.uniform(1 - gamma / 100, 1 + gamma / 100)
    scale_y = np.random.uniform(1 - gamma / 100, 1 + gamma / 100)

    # rotates image based on beta
    # shears image based on beta
    # scales image by scale_x and scale_y
    datagen = ImageDataGenerator(
      rotation_range=beta,
      shear_range=np.tan(np.radians(beta)),
      zoom_range=[scale_x, scale_y]
    )
    
    X_augmented[i] = datagen.random_transform(X[i])

  X_augmented = X_augmented.reshape(-1, 784)
  # augmented_data = np.column_stack((y, X_augmented))
  return pandas.DataFrame(X_augmented)


def render_mnist_augmentation(data, augmented_x, y, amount_images):
  X = data.iloc[:, 1:].values / 255

  augmented_data = augmented_data.iloc[:, 1:].values / 255

  fig, axes = plt.subplots(2, amount_images, figsize=(15, 6))

  for idx, ax in zip(range(amount_images), axes[0]):
    ax.imshow(X[idx].reshape(28, 28), cmap='gray')
    ax.axhline(color='red', y = 9)
    ax.axhline(color='red', y = 18)

    ax.axvline(color='red', x = 9)
    ax.axvline(color='red', x = 18)
    ax.set_title(f"Original: {y[idx]}")
    ax.axis('off')

  for idx, ax in zip(range(amount_images), axes[1]):
    ax.imshow(augmented_data[idx].reshape(28, 28), cmap='gray')
    ax.axhline(color='red', y = 9)
    ax.axhline(color='red', y = 18)

    ax.axvline(color='red', x = 9)
    ax.axvline(color='red', x = 18)
    ax.set_title(f"Augmented: {y[idx]}")
    ax.axis('off')

  plt.suptitle('Original and Augmented Images')
  plt.show()
