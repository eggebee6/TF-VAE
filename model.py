import tensorflow as tf

class VAEModel(tf.keras.Model):
  IMAGE_HEIGHT = 28
  IMAGE_WIDTH = 28

  LATENT_DIMS = 2

  CONV_STRIDE = 2
  CONV_FILTER_SIZE = 3
  NUM_CONV_FILTERS_1 = 32
  NUM_CONV_FILTERS_2 = 64

  TOTAL_DOWNSAMPLING = 4
  DOWNSAMPLED_HEIGHT = IMAGE_HEIGHT / TOTAL_DOWNSAMPLING
  DOWNSAMPLED_WIDTH = IMAGE_WIDTH / TOTAL_DOWNSAMPLING
  LATENT_SHAPE = (DOWNSAMPLED_HEIGHT * DOWNSAMPLED_WIDTH, )

  def __init__(self):
    super(VAEModel, self).__init__()

    # Create encoder
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = (28, 28, 1)),
      tf.keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        strides = (2, 2),
        activation = 'relu'),
      tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        strides = (2, 2),
        activation = 'relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10 * 2)
    ])

    # Create decoder
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = (10, )),
      tf.keras.layers.Dense(
        units = 7 * 7 * 32,
        activation = tf.nn.relu),
      tf.keras.layers.Reshape(target_shape = (7, 7, 32)),
      tf.keras.layers.Conv2DTranspose(
        filters = 64,
        kernel_size = (3, 3),
        strides = (2, 2),
        padding = 'same',
        activation = 'relu'),
      tf.keras.layers.Conv2DTranspose(
        filters = 32,
        kernel_size = (3, 3),
        strides = (2, 2),
        padding = 'same',
        activation = 'relu'),
      tf.keras.layers.Conv2DTranspose(
        filters = 1,
        kernel_size = (3, 3),
        strides = (1, 1),
        padding = 'same')
    ])

    # Create loss metrics
    self.total_loss_metric = tf.keras.metrics.Mean(name = "Loss")


  @property
  def metrics(self):
    return [self.total_loss_metric]


  def call(self, inputs, training = False):
    """ Forward pass through VAE model

    Parameters:
    input - Input images
    training - True if the call is for training, false otherwise

    Outputs:
    recon - Reconstructed input
    latent_sample - Latent sample used for reconstruction
    mean - Encoder mean parameter
    logvars - Encoder log-variances parameter
    """

    # Get distribution parameters from encoder
    latent_params = self.encoder(inputs)
    mean, logvars = tf.split(latent_params, num_or_size_splits = 2, axis = 1)

    # Sample from distribution
    epsilons = tf.random.normal(
      shape = tf.shape(logvars),
      mean = 0.0,
      stddev = 1.0)
    sigmas = tf.math.exp(0.5 * logvars)
    latent_sample = mean + sigmas * epsilons

    # Decode latent sample
    recon = self.decoder(latent_sample)

    return recon, latent_sample, mean, logvars


  def train_step(self, data):
    """ Custom training step

    TODO: Document this
    """
    with tf.GradientTape() as tape:
      recon, latent_sample, mean, logvars = self(data, training = True)

      recon_loss = tf.keras.losses.mean_squared_error(data, recon)
      # TODO: KL divergence

      total_loss = recon_loss
      # TODO: total_loss = recon_loss + kl_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    self.total_loss_metric.update_state(total_loss)

    return {"Total loss": self.total_loss_metric.result()}
