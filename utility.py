import tensorflow as tf

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
