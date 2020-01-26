# Step-by-step example of migrating TensorFlow research code from TF1 &rarr; TF2

[TensorFlow 2.0 is out](), and it contains [some major changes]() compared to TF1. I've struggled to find some really good step-by-step examples of migration efforts on relatively advanced code, so as a part of another post on [generating recommendations with deep contextual bandits with TensorFlow 2.0](), I've written the following. 

In this post I demonstrate how I systematically went through code from TF1, and migrated it to TF2. To help this effort, the TF group has released a [script]() that makes TF1 code compatible with TF2. What this script does is to change the order of arguments - which is important - and append `tf.compat.v1` to many of the methods that have changed. It does not, however, change any of the fundamental syntax, like replacing sessions with functions and introducing eager execution.

To illustrate how big a difference TF2 syntax is from TF1 (from the official [migration site](https://www.tensorflow.org/guide/migrate)). 

TF1:

```python
in_a = tf.placeholder(dtype=tf.float32, shape=(2))
in_b = tf.placeholder(dtype=tf.float32, shape=(2))

def forward(x):
  with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", initializer=tf.ones(shape=(2,2)),
                        regularizer=tf.contrib.layers.l2_regularizer(0.04))
    b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
    return W * x + b

out_a = forward(in_a)
out_b = forward(in_b)

reg_loss = tf.losses.get_regularization_loss(scope="matmul")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outs = sess.run([out_a, out_b, reg_loss],
                feed_dict={in_a: [1, 0], in_b: [0, 1]})
```

TF2:

```python
W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
print(out_a)

out_b = forward([0,1])

regularizer = tf.keras.regularizers.l2(0.04)
reg_loss = regularizer(W)
```

A lot neater, eh? 

## Minor tweaks

You can find the TF1 code in the official TensorFlow research library, or organised a bit differently on my github. The following will use my reorganised code. I won't be using the official migration script, because I want to flush out all the differences, one by one.

```
.
├── linear_bandit
│   ├── __init__.py
│   ├── bandit_algorithm.py
│   ├── bayesian_nn.py
│   ├── contextual_bandit.py
│   ├── contextual_dataset.py
│   ├── linear_full_posterior_sampling.py
│   ├── neural_linear_sampling.py
│   ├── sample_jester_data.py
└── run.py
```

In a TF < 2 environment, running `python -m run` should produce this:

```
Optimal total reward = 633.17.
Total reward from bandit = 82.21.
Reward ratio = 0.12983874788761313
Everything seems to be working OK!
```

while running the code in a TF2 environment will generate some simple errors pertaining to renamed or deprecated APIs. Lets have a look at the two first.

### `tf.gfile.Open` &rarr; `tf.io.gfile`

The first error is a small one. Simply replace line 22 in `sample_jester_data.py 

```python
tf.gfile.Open(file_name, 'rb') as f: 
```

with 

```python
tf.io.gfile.GFile(file_name, 'rb') as f:
```

###`tf.contrib.training.HParams` &rarr; `tensorboard.plugins.hparams`

In the code from the TF research library, `tf.contrib.training.HParams` is used mainly to store parameter values in, but it [can be used to do hyperparameter tuning](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams). 

Simply replace

```python
hparams_linear = tf.contrib.training.HParams(num_actions=num_actions, 
                                             context_dim=context_dim, 
                                             a0=6,
                                             b0=6,
                                             lambda_prior=0.25,
                                             initial_pulls=2)
```

with

```
hparams_linear = {
    "num_actions": num_actions,
    "context_dim": context_dim,
    "a0": 6,
    "b0": 6,
    "lambda_prior": 0.25,
    "initial_pulls": 2}
```

which can be used together with `tensorboard.plugins.hparams import api as hp` and `hp.KerasCallback()` for hyperparamter tuning while fitting. See the [hparams API](hp.KerasCallback(logdir, hparams)) for more info. For now, to keep it as simple as possible, we will just use a dict and call the keys. Use search-and-replace through all files and change entries like `hparams.lambda_prior` to `hparams['lambda_prior']`.

After correting the above, you should get a new error from `neural_bandit_model.py` 

```
AttributeError: module 'tensorflow' has no attribute 'Session'
```

This leads us to the file where the majority of the important differences from TF1 to TF2 are.

## Getting started

### `neural_bandit_model.py` 

A tad over 200 lines of somewhat complicated code allows flexbility around defining a neural network (hence, NN) model that accepts different kinds of optimisers, and can be tuned with a long list of different parameters. By the end of this, we have less than X lines of code.

Before diving into the nitty-griddy details, it's important to keep the overall aim of the code in mind. There are eight methods (and one `__init__()`) in the class `NeuralBanditModel`. Before starting, I will list each method here, and explain shortly what its purpose is.

1. `build_layer()` - returns a fully connected layer with `num_units` of connected units.
2. `forward_pass()` - does a forward pass on the NN, returns a prediction and the updated NN.
3. `build_model()` - builds the actual NN model, sets the optimiser and initiates the graph.
4. `initialise_graph() ` - runs the session.
5. `assign_lr()` - dynamically sets the learning rate as a (decaying) function of time.
6. `select_optimizer()` - returns the `RMSPropOptimizer` optimiser.
7. `create_summaries() ` - prints some useful summary statistics.
8. `train()` - trains the model by running the session.

Most of the important stuff is happening in `build_model()`. Each layer is build with `build_layer()` in the forward pass (`forward_pass()`), which is called when building the model `build_model()`. This happens when the class is initialised. The other methods, running session, selecting the optimisor etc. are also called while building the model. Importantly, `train()` is imported in `neural_linear_sampling.py` and called when updating the model (line 136). We will deal with this later. 

We will be defining the model from scratch using TF2 syntax. This means that both `build_layer()` and `forward_pass()` will be absorbed into the new `build_model()` method. First, lets have a look the method in its entirety.

```python
  def build_model(self):
    """Defines the actual NN model with fully connected layers.

    The loss is computed for partial feedback settings (bandits), so only
    the observed outcome is backpropagated (see weighted loss).
    Selects the optimizer and, finally, it also initializes the graph.
    """

    # create and store the graph corresponding to the BNN instance
    self.graph = tf.Graph()

    with self.graph.as_default():

      # create and store a new session for the graph
      self.sess = tf.Session()

      with tf.name_scope(self.name):

        self.global_step = tf.train.get_or_create_global_step()

        # context
        self.x = tf.placeholder(
            shape=[None, self.hparams["context_dim"]],
            dtype=tf.float32,
            name="{}_x".format(self.name))

        # reward vector
        self.y = tf.placeholder(
            shape=[None, self.hparams["num_actions"]],
            dtype=tf.float32,
            name="{}_y".format(self.name))

        # weights (1 for selected action, 0 otherwise)
        self.weights = tf.placeholder(
            shape=[None, self.hparams["num_actions"]],
            dtype=tf.float32,
            name="{}_w".format(self.name))

        # with tf.variable_scope("prediction_{}".format(self.name)):
        self.nn, self.y_pred = self.forward_pass()
        self.loss = tf.squared_difference(self.y_pred, self.y)
        self.weighted_loss = tf.multiply(self.weights, self.loss)
        self.cost = tf.reduce_sum(self.weighted_loss) / self.hparams["batch_size"]

        if self.hparams["activate_decay"]:
          self.lr = tf.train.inverse_time_decay(
              self.hparams["initial_lr"], self.global_step,
              1, self.hparams["lr_decay_rate"])
        else:
          self.lr = tf.Variable(self.hparams["initial_lr"], trainable=False)

        # create tensorboard metrics
        self.create_summaries()
        self.summary_writer = tf.summary.FileWriter(
            "{}/graph_{}".format(FLAGS.logdir, self.name), self.sess.graph)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), self.hparams["max_grad_norm"])

        self.optimizer = self.select_optimizer()

        self.train_op = self.optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        self.init = tf.global_variables_initializer()

        self.initialize_graph()

```

Because this method is called with a certain frequency (`update_freq()` in `neural_linear_sampling.py`) we need to keep count of what step we are at, so we can show our model the correct batch number. This is what `tf.train.get_or_create_global_step()` does. However, this does not have a implementation in TF2 yet, so we will have to use `tf.compat.v1.train.get_or_create_global_step()` instead.











