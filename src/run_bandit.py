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

from src.contextual_bandit import run_contextual_bandit

from src.linear_full_posterior_sampling import LinearFullPosteriorSampling
from src.neural_bandit_model import NeuralBanditModel
from src.neural_linear_sampling import NeuralLinearPosteriorSampling
from src.sample_jester_data import sample_jester_data

from src.bandit_algorithm import BanditAlgorithm
from src.contextual_bandit import ContextualBandit

import matplotlib.pyplot as plt

DATA_PATH = '/Users/tmo/Data/bandits/'

tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', DATA_PATH + 'logs/', 'Base directory to save output')

FLAGS(sys.argv)


def plot_model_actions(optimal_action_frequencies, model_action_frequencies, time):
 
    oaf_y = optimal_action_frequencies
    maf_y = model_action_frequencies

    oaf_x = np.arange(0, len(maf_y), 1)
    maf_x = np.arange(0, len(oaf_y), 1)

    ind = np.arange(0, len(optimal_action_frequencies)+1)  
    width = 0.35 

    fig, ax = plt.subplots()
    
    rects1 = ax.bar([x for x in oaf_x], oaf_y, width,
                    color='SkyBlue', label='Optimal')
    
    rects2 = ax.bar([x+width for x in maf_x], maf_y, width,
                    color='IndianRed', label='Model')

    ax.set_ylabel('Frequencies')
    ax.set_title('Actions (t={})'.format(time))
    ax.set_xticks(ind)
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8'))
    ax.legend()


def run_bandit(model, hparams, num_contexts, pct_zero=None, plot=True, plot_freq=500):

    dataset, opt_rewards, opt_actions, num_actions, context_dim  = sample_jester_data(
        file_path=DATA_PATH+'jester/jester_data_40jokes_19181users.npy', 
        num_contexts=num_contexts,
        pct_zero=pct_zero,
        shuffle_rows=False, 
        shuffle_cols=False) 

    num_contexts = dataset.shape[0]
    
    h_actions = []
    h_rewards = []

    t0 = time.time()

    # Run the contextual bandit process
    for i in range(num_contexts):

        context = dataset[i, :context_dim] # Grab the ith line up until joke 32
        action = model.action(context) # Just one model with an action for the context
        reward = dataset[i, context_dim+action] # Grab the reward from the 8 possible rewards

        model.update(context, action, reward)

        h_actions.append(action)
        h_rewards.append(reward)
         
        optimal_action_frequencies = [[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)]
        model_action_frequencies = [[elt, list(h_actions).count(elt)] for elt in set(h_actions)]

        oaf_array = np.array([x[1] for x in optimal_action_frequencies])
        maf_array = np.array([x[1] for x in model_action_frequencies]) 

        if plot and model.t % plot_freq == 0:    

            plot_model_actions(oaf_array, 
                               maf_array, 
                               model.t)
    
    t1 = time.time()
    
    print('Ran {} iterations in {} seconds'.format(num_contexts, t1-t0)) 

    return oaf_array, maf_array, h_rewards, h_actions
