# Ecommerce recommendations with deep contextual bandits using TensorFlow 

Tentative titles:

* Cold start recommendations with Contextual Bandits
* Generating recommendations with contextual bandits in TensorFlow 2.0.
* Bayesian contextual bandits in Tensowflow 2.0

## Abstract

In this post I shortly introduce the concept of a multi-armed bandit, explain its potential commercial use and touch upon its theoretical foundation with a minimum of math. Following this, I demonstrate how to implement a Bayesian version of a multi-armed contextual bandit based on [TensorFlow research code](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits) . At the end, I discuss the steps needed to implement this as a generic solution to making recommendations. This post will be followed by a detailed walk-through demonstrating how to migrate the TF1 code used here to TF2 syntax.

## Introduction

The term *multi-armed bandit* originates from the Las Vegas style slot machines. A hopeful gambler chooses a machine, pulls the arm, and if they're in luck, a bucketful of quaters will sprew out of the machine. Since we're talking Vegas, the bandit part of the name should be self-explanatory. So, standing in front of all the many different colorful machines, which arm should you choose to pull? If the gambler could pull each arm many times, she might be able to learn what machines pay out the most reward. In a broad sense, this is the problem we're trying to model in the following.

We can easily apply the above dilemma to a commercial setting. What commercials or products to do we display? What price to set for this and that user, etc. Like the multi-armed bandit, each user that comes to a website or uses an application, has some probability of taking one or several actions that we find rewarding. This could be subscribing to a newsletter, looking at specific products, or buying something from the webshop. As you might be able to tell, recommendations are just a way to frame an action, and this framework can be seen as generating actions, that in a commercial setting, is simply a recommendation.

At this point, you might be asking yourself: "*What about the **contextual** part of contextual multi-armed bandits?*". As I explain in more technical detail below, the context can be any data that describes the situation we are trying to model in more detail. It can be the users age, previous recommendations, or perhaps even the weather - if that makes sense. In short, what data scientists sometimes to as features. This extra information from the context is embedded in the bandit, and helps to improve the quality of the action we choose. 

Recommendations can we done in different ways, many of which does not require any machine learning. There are many excellent articles that talk about simple - no machine learning - recommendation, or how recommendation fits into personalisation. However, the focus of this article will specifically be on contextual bandits for a few reasons.

1. **Its flexible**. As you will see in the following, contextual multi-armed bandits can be used to solve many machine learning problems that are not nessesarily related to recommendation. The official TensorFlow research library has a few good examples of this. 
2. **The math is relatively uncomplicated**. Naturally, this can be made arbitrarily difficult, but the main components in a multi-armed bandit are easy to understand, and making it "context aware" is also a simple addition to the model. 
3. **A bandit does not nessesarily need any data to make recommendations**. Sometimes referred to as the *cold-start problem*, a bandit does not need any data to start making recommendations. Those recommendations wont nessesarily be very good at the beginning, but with some data, it quickly improves and becomes better than random.

## The contextual multi-armed bandit

A contextual bandit is a special case of a general reinforcement learning problem, the major difference being that the state does not persist. This means that at each step, the agent is presented with a context $x$ and chooses the action $a$ from the set of possible actions $K$. Each different action yield a reward $r$. 

(Insert illustration)

The aim of the agent is to build a model of the distribution of rewards given the context and action. In math
$$
P(r|x,a,\bold{w})
$$
where $\bold w$ is the weights of the model, i.e. a neural network or linear regression model. 

Lets say we start our simulation, and the agent picks up an action that yields the reward 10. Why not just stay in that state, and exploit the safe reward every single step? Well, that sounds nice, but what we if we can get twice as much by looking a bit further? Ideally, we want our agent to balance exploration and exploitation just right, such that our agent is constantly looking for new options, but also knows when to stay put and reap the juicy rewards. This is called the exploration-exploitation dilemma, and is a fundamental crux in reinforcement learning. 

One simple, yet effective, way of solving the exploration-exploitation dilemma is called Thompson sampling (see [Chapelle & Li][An Empirical Evaluation of Thompson Sampling] for details). This idea is this: Each step, draw a new set of parameters from the posterior distribution and pick the action with the highest expected reward given those parameters, then rinse and repeat. Simple right? This has the neat consequence that probable parameters will be drawn more often, and thus refuted or confirmed faster. But importantly,  it also means that our model will still sample unlikely parameter values which is important for exploration. You can see this as a kind of stochastic hypothesis testing. 

You can find several implementations ([here]() or [here]()) of simple multi-armed bandits, and [here]() is a nicely illustrated example of a contextual bandit as well. Those implementations are great for understanding the mechanics, but they are too slow and inflexible for application to any real world data sets. 

## Getting started

In the following, I show code adapted from the TensorFlow research library, [deep contextual bandits](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits). To keep things simple, I've chosen to focus on the neural linear model, and to do something novel, I will apply it to a commercially relevant dataset - the [Retailrocket Ecommerce dataset from Kaggle](https://www.kaggle.com/retailrocket/ecommerce-dataset). As I will explain in a bit more detail below, this dataset requires thinking a bit about the reward structure - but more on that later.

Our overarching aim is to make a contextual bandit that generates recommendations based on the context that the current user is in. 

## Coming up with a meaningful reward structure

From the Kaggle description:

> *The behaviour data, i.e. events like clicks, add to carts, transactions, represent interactions that were collected over a period of 4.5 months. A visitor can make three types of events, namely “view”, “addtocart” or “transaction”. In total there are 2 756 101 events including 2 664 312 views, 69 332 add to carts and 22 457 transactions produced by 1 407 580 unique visitors. For about 90% of events corresponding properties can be found in the “item_properties.csv” file.*

I won't go into too much detail with the data, as that is not the purpose of the article. However, as you will find out, if you look through the RetailRocket notebook in my git, the dataset is substantial with some 2.7 million events distributed over 1.4 million unique visitors, a bit less than 12000 transactions, and some 235000 unique items.

Massaging a real world dataset to a point where reinforcement learning makes sense, is often a creative challenge that requires a lot of experimentation and no rules are written in stone. Because the agent will learn to maximise rewards, building this space that the agent traverses in order to learn its policy, is key to generating actions (in our case recommendations) that make sense.

### Reducing the reward space with unsupervised clustering

Recall that we have two dimensions to our reward matrix: Users and items. Each user can potentially be recommended each unique item. If we use the original data with `r=1` at each transaction event, this will result in a `NxM` matrix with 1.4 million rows and 235000 columns. This results in a search space of approximately 329 billion combinations of users and items, with only  12000 reward events. As you might guess, this wont really work well. 

First we are going to remove a large part of the data in the tails of the item and visitor dimension (see the notebook on details). Then, we are going to make two clustering models: One for events, and another for items. When a visitor enters our site, they generate an event: `view`, `addtocart` or `transaction`. Given the event context (visitorid, number of visits, cart adds and transactions), we want to recommend an item.



 we retrieve information about their cluster from a cookie, or if they are first-time visitors, we perform inference on our clustering model to get their cluster. Armed with their cluster, we ask our bandit what the most rewarding action is. We apply the same logic to our items: The bandit learns not to choose a "item action", but instead a "cluster action". Each cluster contains a number of items that we can randomise between showing. A advanced alternative to this is shortly discussed in the next steps section.

To make things manageble and still retain some some diversity, I have chosen to seperate 



### Rewards we care about

To make meaningful recommendations, we need to ask ourselves what we want. We like transactions best, so that should yield the highest reward. We also like that users add items to their cart, and when they visit the same item many times, as both of them correlate with subsequent transaction events. 

```python
def rewards(row):
    
    v = 0
    
    if row['event'] == 'transaction':
        v = 15
    
    if row['event'] == 'addtocart':
        v = 10
    
    if row['event'] == 'view' and row['no_view'] >= 2:
        v = 5
        
    return v
```

This small function implements our reward structure. Basically, this says that any transaction yields 15 reward, any add to cart action 10, and if a specific user revisits the same item twice, we give that action 5 reward.

## TensorFlow implementation



[An Empirical Evaluation of Thompson Sampling]: https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf



