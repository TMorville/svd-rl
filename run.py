from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from absl import app
from absl import flags

from linear_bandit.sample_jester_data import sample_jester_data
from linear_bandit.sample_retail_data import sample_retail_data

from linear_bandit.contextual_bandit import run_contextual_bandit

from linear_bandit.linear_full_posterior_sampling import LinearFullPosteriorSampling

from linear_bandit.bandit_algorithm import BanditAlgorithm
from linear_bandit.contextual_bandit import ContextualBandit

import matplotlib.pyplot as plt

data_route = '/Users/tmo/Data/bandits/'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', data_route + 'logs/', 'Base directory to save output')

FLAGS(sys.argv)


def get_jester_data(num_contexts=None):

    num_actions = 8
    context_dim = 32
    
    # 40 jokes. Eight for the arms, and 32 as context for the user.
    
    num_contexts = min(19181, num_contexts)
    
    # 19181 users had ratings for all 40 jokes.
    
    file_name = '/Users/tmo/Data/bandits/jester_data_40jokes_19181users.npy'
    dataset, opt_jester = sample_jester_data(file_name, context_dim,
                                             num_actions, num_contexts,
                                             shuffle_rows=True,
                                             shuffle_cols=True)
    opt_rewards, opt_actions = opt_jester
    
    return dataset, opt_rewards, opt_actions, num_actions, context_dim


sampled_vals = get_jester_data(100)

dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals

hparams_linear = {
    "num_actions": num_actions,
    "context_dim": context_dim,
    "a0": 6,
    "b0": 6,
    "lambda_prior": 0.25,
    "initial_pulls": 2}

linear_bandit = LinearFullPosteriorSampling(name='linear_bandit', hparams=hparams_linear)


def run_bandit(model, hparams, plot=True):
        
    num_contexts = dataset.shape[0]
    
    h_actions = []
    h_rewards = []
    
    # Run the contextual bandit process
    for i in range(num_contexts):
        context = dataset[i, :context_dim] # Grab the ith line up until joke 32
        action = model.action(context) # Just one model with an action for the context
        reward = dataset[i, context_dim+action] # Grab the reward from the 8 possible rewards

        model.update(context, action, reward)

        h_actions.append(action)
        h_rewards.append(reward)
        
        if plot and model.t % 500 == 0:
            optimal_action_frequencies = [[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)]
            model_action_frequencies = [[elt, list(h_actions).count(elt)] for elt in set(h_actions)]
            
            plot_optimal_model_actions(optimal_action_frequencies, 
                                       model_action_frequencies, 
                                       model.t)
            
        
    print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
    print('Total reward from bandit = {}.'.format(np.sum(h_rewards)))
    print('Reward ratio = {}'.format(np.sum(h_rewards)/np.sum(opt_rewards)))
        
    optimal_action_frequencies = [[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)]
    model_action_frequencies = [[elt, list(h_actions).count(elt)] for elt in set(h_actions)]
    
    return optimal_action_frequencies, model_action_frequencies

if __name__ == '__main__':
    oaf, maf = run_bandit(linear_bandit, hparams_linear, plot=False)

    if oaf and maf:
        print("Everything seems to be working OK!")
