"""Learn vote ideal points for Senators using variational inference.

We use the following Bayesian model, where `i` indexes Senators and `j` indexes
bills:

x_i ~ N(0, 1)
alpha_i, eta_j ~ N(0, 1)
v_ij ~ Bernoulli(sigma(alpha_j + x_i * eta_j)),

where v_ij is the indicator of `i`'s vote on bill `j` and `sigma` is the 
inverse-logit function. We refer to `x` as the ideal point, `alpha` as the
popularity, and `eta` as the polarity.

To preprocess data, make sure to run `preprocess_senate_votes.py` before 
running.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
import sys
import warnings

# Dependency imports
from absl import flags
from absl import app
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Adam learning rate.")
flags.DEFINE_integer("max_steps",
                     default=2500,
                     help="Number of training steps to run.")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples to use for ELBO approximation.")
flags.DEFINE_integer("senate_session",
                     default=114,
                     help="Senate session.")
flags.DEFINE_integer("print_steps",
                     default=100,
                     help="Number of steps to print and save results.")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
FLAGS = flags.FLAGS


def build_input_pipeline(data_dir):
  """Load data and build iterator for minibatches.
  
  Args:
    data_dir: The directory where the data is located. There must be four
      files inside the rep: `votes.npy`, `bill_indices.npy`, 
      `senator_indices.npy`, and `senator_map.txt`. `votes_npy` is a binary
      vector with shape [num_total_votes], indicating the outcome of each cast
      vote. `bill_indices.npy` is a vector with shape [num_total_votes], where
      each entry is an integer in {0, 1, ..., num_bills - 1}, indicating the
      bill index voted on. `senator_indices.npy` is a vector with shape 
      [num_total_votes], where each entry is an integer in {0, 1, ..., 
      num_senators - 1}, indicating the Senator voting. Finally, 
      `senator_map.txt` is a list of each Senator's name.
  """
  votes = np.load(os.path.join(data_dir, "votes.npy"))
  bill_indices = np.load(os.path.join(data_dir, "bill_indices.npy"))
  senator_indices = np.load(os.path.join(data_dir, "senator_indices.npy"))
  senator_map = np.loadtxt(os.path.join(data_dir, "senator_map.txt"), dtype=str)
  num_bills = len(np.unique(bill_indices))
  num_senators = len(senator_map)
  dataset_size = len(votes)
  dataset = tf.data.Dataset.from_tensor_slices(
      (votes, bill_indices, senator_indices))
  # Use the complete dataset as a batch.
  batch_size = len(votes)
  batches = dataset.repeat().batch(batch_size).prefetch(batch_size)
  iterator = iter(batches)
  return iterator, senator_map, num_bills, num_senators, dataset_size


def print_ideal_points(ideal_point_loc, senator_map):
    """Order and print ideal points for Tensorboard.

    Args:
        ideal_point_loc: tf.Variable or tf.Tensor of shape [num_senators]
        senator_map: list of strings of length num_senators

    Returns:
        A string with senator names ordered by their ideal points
    """
    # Convert ideal points to NumPy array if necessary
    if isinstance(ideal_point_loc, tf.Variable) or isinstance(ideal_point_loc, tf.Tensor):
        ideal_point_array = ideal_point_loc.numpy()
    else:
        ideal_point_array = np.array(ideal_point_loc)

    # Ensure senator_map is a list of strings
    senator_map = [str(s) for s in senator_map]

    # Get sorted indices and return ordered senator names
    sorted_indices = np.argsort(ideal_point_array)
    sorted_senators = [senator_map[i] for i in sorted_indices]

    return ", ".join(sorted_senators)


def get_log_prior(samples):
    """Return log prior of sampled Gaussians.

    Args:
        samples: A tf.Tensor with shape [num_samples, latent_dim]

    Returns:
        log_prior: A tf.Tensor with shape [num_samples], log prior summed across latent dim
    """
    prior_distribution = tfp.distributions.Normal(loc=0., scale=1.)
    log_prior = tf.reduce_sum(prior_distribution.log_prob(samples), axis=1)
    return log_prior


def get_entropy(distribution, samples):
  """Return entropy of sampled Gaussians.
  
  Args:
    samples: A Tensor with shape [num_samples, :].
  
  Returns:
    entropy: A `Tensor` with shape [num_samples], with the entropy
      summed across the latent dimension.
  """
  entropy = -tf.reduce_sum(distribution.log_prob(samples), axis=1)
  return entropy


def get_elbo(votes,
             bill_indices,
             senator_indices,
             ideal_point_distribution,
             polarity_distribution,
             popularity_distribution,
             dataset_size,
             num_samples):
  """Approximate ELBO using reparameterization.
  
  Args:
    votes: A binary vector with shape [batch_size].
    bill_indices: An int-vector with shape [batch_size].
    senator_indices: An int-vector with shape [batch_size].
    ideal_point_distribution: A Distribution object with parameter shape 
      [num_senators].
    polarity_distribution: A Distribution object with parameter shape 
      [num_bills].
    popularity_distribution: A Distribution object with parameter shape 
      [num_bills]
    dataset_size: The number of observations in the total data set (used to 
      calculate log-likelihood scale).
    num_samples: Number of Monte-Carlo samples.
  
  Returns:
    elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value is
      averaged across samples and summed across batches.
  """
  ideal_point_samples = ideal_point_distribution.sample(num_samples)
  polarity_samples = polarity_distribution.sample(num_samples)
  popularity_samples = popularity_distribution.sample(num_samples)
  
  ideal_point_log_prior = get_log_prior(ideal_point_samples)
  polarity_log_prior = get_log_prior(polarity_samples)
  popularity_log_prior = get_log_prior(popularity_samples)
  log_prior = ideal_point_log_prior + polarity_log_prior + popularity_log_prior

  ideal_point_entropy = get_entropy(ideal_point_distribution, 
                                    ideal_point_samples)
  polarity_entropy = get_entropy(polarity_distribution, polarity_samples)
  popularity_entropy = get_entropy(popularity_distribution, popularity_samples)
  entropy = ideal_point_entropy + polarity_entropy + popularity_entropy

  selected_ideal_points = tf.gather(ideal_point_samples, 
                                    senator_indices, 
                                    axis=1)
  selected_polarities = tf.gather(polarity_samples, bill_indices, axis=1) 
  selected_popularities = tf.gather(popularity_samples, bill_indices, axis=1) 
  vote_logits = (selected_ideal_points * 
                 selected_polarities + 
                 selected_popularities)

  vote_distribution = tfp.distributions.Bernoulli(logits=vote_logits)
  vote_log_likelihood = vote_distribution.log_prob(votes)
  vote_log_likelihood = tf.reduce_sum(vote_log_likelihood, axis=1)
  
  elbo = log_prior + vote_log_likelihood + entropy
  elbo = tf.reduce_mean(elbo)

  tf.summary.scalar("elbo/elbo", elbo)
  tf.summary.scalar("elbo/log_prior", tf.reduce_mean(log_prior))
  tf.summary.scalar("elbo/vote_log_likelihood", 
                    tf.reduce_mean(vote_log_likelihood))
  tf.summary.scalar("elbo/entropy", tf.reduce_mean(entropy))
  return elbo


def main(argv):
    del argv
    tf.random.set_seed(FLAGS.seed)

    # Directories
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    source_dir = os.path.join(project_dir, f"data/senate-votes/{FLAGS.senate_session}")
    data_dir = os.path.join(source_dir, "clean")
    save_dir = os.path.join(source_dir, "fits")
    os.makedirs(save_dir, exist_ok=True)

    # Data
    iterator, senator_map, num_bills, num_senators, dataset_size = build_input_pipeline(data_dir)
    votes, bill_indices, senator_indices = next(iterator)

    # Variational parameters
    ideal_point_loc = tf.Variable(tf.zeros([num_senators]), name="ideal_point_loc")
    ideal_point_logit = tf.Variable(tf.zeros([num_senators]), name="ideal_point_logit")
    ideal_point_scale = tf.nn.softplus(ideal_point_logit)

    polarity_loc = tf.Variable(tf.zeros([num_bills]), name="polarity_loc")
    polarity_logit = tf.Variable(tf.zeros([num_bills]), name="polarity_logit")
    polarity_scale = tf.nn.softplus(polarity_logit)

    popularity_loc = tf.Variable(tf.zeros([num_bills]), name="popularity_loc")
    popularity_logit = tf.Variable(tf.zeros([num_bills]), name="popularity_logit")
    popularity_scale = tf.nn.softplus(popularity_logit)

    # TensorBoard writer
    writer = tf.summary.create_file_writer(save_dir)

    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    trainable_vars = [ideal_point_loc, ideal_point_logit,
                      polarity_loc, polarity_logit,
                      popularity_loc, popularity_logit]

    for step in range(FLAGS.max_steps):
        start_time = time.time()
        with tf.GradientTape() as tape:
            # Build distributions
            ideal_point_dist = tfp.distributions.Normal(loc=ideal_point_loc,
                                                        scale=ideal_point_scale)
            polarity_dist = tfp.distributions.Normal(loc=polarity_loc,
                                                     scale=polarity_scale)
            popularity_dist = tfp.distributions.Normal(loc=popularity_loc,
                                                       scale=popularity_scale)
            # Compute ELBO
            elbo_val = get_elbo(votes, bill_indices, senator_indices,
                                ideal_point_dist, polarity_dist, popularity_dist,
                                dataset_size, FLAGS.num_samples)
            loss = -elbo_val

        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        duration = time.time() - start_time

        # Print info
        if step % FLAGS.print_steps == 0:
            print(f"Step {step:>4d}, ELBO: {elbo_val:.3f} ({duration:.3f} sec)")

            # Write TensorBoard summaries
            with writer.as_default():
                tf.summary.scalar("ELBO", elbo_val, step=step)
                # Print ordered ideal points
                ideal_point_names = print_ideal_points(ideal_point_loc, senator_map)
                tf.summary.text("Ideal Points", ideal_point_names, step=step)

        # Save parameters every 100 steps
        if step % 100 == 0 or step == FLAGS.max_steps - 1:
            param_save_dir = os.path.join(save_dir, "params")
            os.makedirs(param_save_dir, exist_ok=True)

            np.save(os.path.join(param_save_dir, "ideal_point_loc"), ideal_point_loc.numpy())
            np.save(os.path.join(param_save_dir, "ideal_point_scale"), ideal_point_scale.numpy())
            np.save(os.path.join(param_save_dir, "polarity_loc"), polarity_loc.numpy())
            np.save(os.path.join(param_save_dir, "polarity_scale"), polarity_scale.numpy())
            np.save(os.path.join(param_save_dir, "popularity_loc"), popularity_loc.numpy())
            np.save(os.path.join(param_save_dir, "popularity_scale"), popularity_scale.numpy())

if __name__ == "__main__":
    app.run(main)