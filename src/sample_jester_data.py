import tensorflow as tf

import numpy as np


def sample_jester_data(file_path, 
                       num_contexts,
                       pct_zero=None,
                       shuffle_rows=True, 
                       shuffle_cols=False):
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

    num_actions = 8
    context_dim = 32
    
    # 40 jokes. Eight for the arms, and 32 as context for the user.
    
    with tf.gfile.Open(file_path, 'rb') as f:
        dataset = np.load(f)

    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    
    if shuffle_rows:
        np.random.shuffle(dataset)
        
    dataset = dataset[:num_contexts, :]
    
    if pct_zero > 0:
 
        remove_abs = int(round(dataset.size * pct_zero))

        mask=np.zeros(dataset.size,dtype=bool)
        mask[:remove_abs] = True
        np.random.shuffle(mask)
        mask=mask.reshape(dataset.shape)

        dataset[mask] = 0

    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a]
                          for i, a in enumerate(opt_actions)])

    return dataset, opt_rewards, opt_actions,  num_actions, context_dim
