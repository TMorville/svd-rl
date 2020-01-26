import tensorflow as tf

import numpy as np

def sample_jester_data(file_name, context_dim, num_actions, num_contexts,
                       shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.

    Args:
    file_name: Route of file containing the modified Jester dataset.
    context_dim: Context dimension (i.e. vector with some ratings from a user).
    num_actions: Number of actions (number of joke ratings to predict).
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    shuffle_cols: Whether or not context/action jokes are randomly shuffled.

    Returns:
    dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """

    with tf.gfile.Open(file_name, 'rb') as f:
        dataset = np.load(f)

    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
        dataset = dataset[:num_contexts, :]

    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a]
                          for i, a in enumerate(opt_actions)])

    return dataset, (opt_rewards, opt_actions)
