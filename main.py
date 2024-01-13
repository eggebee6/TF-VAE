import tensorflow as tf
import numpy as np
import utility as utils
import model
import pathlib
import datetime

print()
print("Starting...")

# Set paths to MNIST images
MNIST_IMAGE_PATH = pathlib.Path('C:/Projects/Datasets/MNIST')
MNIST_TRAINING_PATH = MNIST_IMAGE_PATH / 'training'
MNIST_VALIDATION_PATH = MNIST_IMAGE_PATH / 'testing'

# Set up TensorBoard
time_now = datetime.datetime.now()
TENSORBOARD_LOG_DIR = './logs/' + time_now.strftime('%m%d%H%M')

file_writer = tf.summary.create_file_writer(TENSORBOARD_LOG_DIR)
file_writer.set_as_default()

# Create training and validation datastores
ds_train = utils.initialize_training_datastore(MNIST_TRAINING_PATH)
ds_validate = utils.initialize_validation_datastore(MNIST_VALIDATION_PATH)

# Create VAE
model = model.VAEModel()
model.compile(optimizer = "adam")

# Train model
epoch_count = 5
model.fit(ds_train, epochs = epoch_count)

# Display results
for imgs in ds_validate.take(1):
  recon_imgs, _, _, _ = model(imgs)
  tf.summary.image("Input images", imgs[0:9], step = 0)
  tf.summary.image("Recon images", recon_imgs[0:9], step = 0)
