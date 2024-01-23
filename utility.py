import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_MINIBATCH_SIZE = 64
DEFAULT_IMAGE_SIZE = (28, 28)

def preprocess_images(ds):
  return ds.map(lambda img: tf.cast(img, tf.float32) / 255.0)

def initialize_training_datastore(
  ds_path,
  minibatch_size = DEFAULT_MINIBATCH_SIZE):
  """Initialize the training datastore"""

  ds = tf.keras.utils.image_dataset_from_directory(
    directory = ds_path,
    labels = 'inferred',
    label_mode = None,
    color_mode = 'grayscale',
    batch_size = minibatch_size,
    image_size = DEFAULT_IMAGE_SIZE,
    shuffle = True,
    validation_split = 0,
  )

  ds = ds.apply(preprocess_images)
  ds = ds.shuffle(minibatch_size * 2)
  return ds


def initialize_validation_datastore(
  ds_path,
  minibatch_size = DEFAULT_MINIBATCH_SIZE):
  """Initialize the validation datastore"""

  ds = tf.keras.utils.image_dataset_from_directory(
    directory = ds_path,
    labels = 'inferred',
    label_mode = None,
    color_mode = 'grayscale',
    batch_size = minibatch_size,
    image_size = DEFAULT_IMAGE_SIZE,
    shuffle = True,
    validation_split = 0,
  )

  ds = ds.apply(preprocess_images)
  ds = ds.shuffle(minibatch_size * 2)
  return ds


def plot_latent_plane(model,
  grid_lims = [-1.5, 1.5],
  side_count = 25):
  
  # Set up latent space grid
  grid_x = np.linspace(grid_lims[0], grid_lims[1], side_count)
  grid_y = np.linspace(grid_lims[0], grid_lims[1], side_count)

  # Initialize canvas
  canvas_shape = (28 * side_count, 28 * side_count)
  canvas = np.zeros(canvas_shape)

  # Decode images from each point in the grid
  for i, x in enumerate(grid_x):
    for j, y in enumerate(grid_y):
      # Decode image from latent grid point
      latent_point = tf.constant([x, y], shape = (1, 2))
      img = model.decoder(latent_point)
      img = tf.reshape(img, [28, 28])

      # Add image to canvas
      canvas_x = x - grid_lims[0]
      canvas_y = y - grid_lims[0]
      canvas[i * 28 : (i+1) * 28, j * 28 : (j+1) * 28] = img.numpy()

  # Plot canvas
  plt.figure(figsize = (5, 5))
  plt.imshow(canvas, cmap = 'Greys_r')
  plt.axis('Off')
  plt.show()
