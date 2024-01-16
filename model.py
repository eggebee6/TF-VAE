import tensorflow as tf

class VAEModel(tf.keras.Model):
  IMAGE_HEIGHT = 28
  IMAGE_WIDTH = 28

  LATENT_DIMS = 10

  CONV_STRIDE = 2
  CONV_FILTER_SIZE = 3
  NUM_CONV_FILTERS_1 = 32
  NUM_CONV_FILTERS_2 = 64

  TOTAL_DOWNSAMPLING = 4
  DOWNSAMPLED_HEIGHT = IMAGE_HEIGHT / TOTAL_DOWNSAMPLING
  DOWNSAMPLED_WIDTH = IMAGE_WIDTH / TOTAL_DOWNSAMPLING
  LATENT_SHAPE = (DOWNSAMPLED_HEIGHT * DOWNSAMPLED_WIDTH, )

  def __init__(self,
    recon_param_scale = 1.0,
    kl_param_scale = 1.0e-3):

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
      tf.keras.layers.Dense(VAEModel.LATENT_DIMS * 2)
    ])

    # Create decoder
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = (VAEModel.LATENT_DIMS, )),
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

    # Set up loss functions
    self.recon_loss_fn = tf.keras.losses.MeanSquaredError()

    self.kl_loss_fn = lambda mean, logvars: 0.5 * tf.math.reduce_sum(
      tf.math.exp(logvars) + tf.math.square(mean) - 1 - logvars,
      axis = 1)

    # Initialize training parameters
    self.recon_param_scale = tf.constant(recon_param_scale)
    self.kl_param_scale = tf.constant(kl_param_scale)

    self.epoch_scale = tf.Variable(0.0, trainable = False)

    self.min_recon_loss = tf.Variable(10000.0, trainable = False)
    self.kl_balance_scale = tf.Variable(0.0, trainable = False)

    # Create loss metrics
    self.total_loss_metric = tf.keras.metrics.Mean(name = "Loss")
    self.recon_loss_metric = tf.keras.metrics.Mean(name = "Recon loss")
    self.kl_loss_metric = tf.keras.metrics.Mean(name = "KL loss")


  @property
  def metrics(self):
    return [
      self.total_loss_metric,
      self.recon_loss_metric,
      self.kl_loss_metric]


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


  @tf.function
  def train_step(self, data):
    """ Custom training step

    TODO: Document this
    """
    with tf.GradientTape() as tape:
      # Forward pass through model
      recon, latent_sample, mean, logvars = self(data, training = True)

      # Calculate losses
      recon_loss = self.recon_loss_fn(data, recon)
      kl_loss = self.kl_loss_fn(mean, logvars) * self.kl_balance_scale * self.epoch_scale

      # Weight losses and sum for total loss
      recon_loss = recon_loss * self.recon_param_scale
      kl_loss = kl_loss * self.kl_param_scale
      total_loss = recon_loss + kl_loss

    # Adjust KL balance
    if (recon_loss > 0):
      if (recon_loss < self.min_recon_loss):
        self.min_recon_loss.assign(recon_loss)
      self.kl_balance_scale.assign((self.min_recon_loss / recon_loss))

    # Get gradients and update model
    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Update metrics
    self.total_loss_metric.update_state(total_loss)
    self.recon_loss_metric.update_state(recon_loss)
    self.kl_loss_metric.update_state(kl_loss)

    return {
      "Total loss": self.total_loss_metric.result(),
      "Recon loss": self.recon_loss_metric.result(),
      "KL loss" : self.kl_loss_metric.result()}


class VAECallback(tf.keras.callbacks.Callback):
  def __init__(self, vae, total_epochs):
    if (total_epochs < 1):
      raise ValueError("Total epochs must be at least 1")

    self.vae = vae
    self.total_epochs = total_epochs

  def on_epoch_begin(self, epoch, logs = None):
    # Adjust epoch scale at start of each epoch (first epoch is 0)
    self.vae.epoch_scale.assign(epoch / self.total_epochs)
