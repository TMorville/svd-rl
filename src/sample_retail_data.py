import tensorflow as tf

import numpy as np
import pandas as pd

def sample_retail_data(file_name, context_dim, num_actions, num_contexts,
                       shuffle_rows=True, shuffle_cols=False, cutoff=None):
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

    retail_df = pd.read_csv(file_name)

    most_freq_items = retail_df['itemid'].value_counts().loc[lambda x : x>cutoff]
    most_visited = retail_df.loc[retail_df['itemid'].isin(most_freq_items.index)]

    print("{} items, {} visitors".format(most_visited.itemid.nunique(), most_visited.visitorid.nunique()))

    most_visited_matrix = most_visited.pivot_table(index='visitorid', columns='itemid', values='reward').as_matrix()
    
    dataset = np.nan_to_num(most_visited_matrix)

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
