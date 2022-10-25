#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:32:16 2022

@author: Jorgen Svane
"""

import os
import numpy as np
import ray
from ray import air, tune
import ray.rllib.algorithms.ppo as ppo

ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["env"] = "CartPole-v0"
config["num_gpus"] = 0
config["num_workers"] = 1
config["framework"] = "tf2"
config["eager_tracing"] = True
config["lr"] = tune.grid_search([0.01, 0.001, 0.0001])

# fist tuning to actually get a trained agent
results = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"training_iteration": 30,
                                    "episode_reward_mean": 200},
                              local_dir="ppo_test"),
    param_space=config
    )

results.fit()


# Get a dataframe for the last reported results of all of the trials
df = results.get_dataframe()


# reload the previous experiment
from ray.tune import ExperimentAnalysis
analysis = ExperimentAnalysis("./ppo_test/PPO")

best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max") 

best_ckp = analysis.get_best_checkpoint(best_trial,metric="episode_reward_mean", mode="max")

config["lr"] = 0.0 # the below can't handle tune.grid_search and also a change to config

agent = ppo.PPO(config=config, env= "CartPole-v0")
agent.restore(best_ckp)

policy = agent.get_policy()
fcn = policy.model # instance of ray.rllib.models.tf.fcnet.FullyConnectedNetwork
fcn.base_model._name = "MyDefaultFcn" # give the model a more descriptive name
fcn.base_model.summary()

"""

Model: "MyDefaultFcn"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 observations (InputLayer)      [(None, 4)]          0           []                               
                                                                                                  
 fc_1 (Dense)                   (None, 256)          1280        ['observations[0][0]']           
                                                                                                  
 fc_value_1 (Dense)             (None, 256)          1280        ['observations[0][0]']           
                                                                                                  
 fc_2 (Dense)                   (None, 256)          65792       ['fc_1[0][0]']                   
                                                                                                  
 fc_value_2 (Dense)             (None, 256)          65792       ['fc_value_1[0][0]']             
                                                                                                  
 fc_out (Dense)                 (None, 2)            514         ['fc_2[0][0]']                   
                                                                                                  
 value_out (Dense)              (None, 1)            257         ['fc_value_2[0][0]']             
                                                                                                  
==================================================================================================
Total params: 134,915
Trainable params: 134,915
Non-trainable params: 0
__________________________________________________________________________________________________

"""

fcn.base_model.save_weights("my_weights.h5")

# agent.get_policy().model.base_model.load_weights("my_weights.h5")

# fcn.base_model.save("my_model.h5")

trained_weights = agent.get_policy().get_weights()

new_agent = ppo.PPO(config=config, env= "CartPole-v0")

untrained_weights = new_agent.get_policy().get_weights()

# check whether weights are equal

arrays_equal = all([np.array_equal(t,u) for t,u in zip(trained_weights, untrained_weights)])

print(f"Trained and untrained weights equal? {arrays_equal}") # False

new_agent.get_policy().model.base_model.load_weights("my_weights.h5")

loaded_weights = new_agent.get_policy().get_weights()

arrays_equal = all([np.array_equal(t,l) for t,l in zip(trained_weights, loaded_weights)])

print(f"Trained and loaded weights equal? {arrays_equal}") # True

del agent

class PPOalgo(ppo.PPO):
    def __init__(self, config, **kwargs):
        super(PPOalgo, self).__init__(config, **kwargs)
        """ Needs full path here!"""
        _cwd = os.path.dirname(os.path.abspath(__file__))
        self.get_policy().model.base_model.load_weights(_cwd + "/my_weights.h5")
        self.workers.sync_weights() # Important!!!
        
    def reset_config(self, new_config):
        """ to enable reuse of actors """
        self.config = new_config
        return True    

# second tuning with lr = 0.0 to confirm weights are loaded and not changed
# hence, config changed relative to first tuning
results = tune.Tuner(
    PPOalgo,
    run_config=air.RunConfig(stop={"training_iteration": 1, # note only one iteration
                                   "episode_reward_mean": 200},
                              local_dir="ppo_test"),
    param_space=config
    )

results.fit()

# reload the second tuning results
from ray.tune import ExperimentAnalysis
analysis = ExperimentAnalysis("./ppo_test/PPOalgo_2022-10-24_21-23-42") # put the right name here

best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max") 

best_ckp = analysis.get_best_checkpoint(best_trial,metric="episode_reward_mean", mode="max")

# Now able to continue training in rllib if needed - remember to reset lr to something != 0.0 !!!
# config["lr"] = 1e-5

agent = ppo.PPO(config=config, env= "CartPole-v0")
agent.restore(best_ckp)

weights = agent.get_policy().get_weights()

arrays_equal = all([np.array_equal(t,w) for t,w in zip(trained_weights, weights)]) # True => no changes as lr = 0.0

print(f"Trained and recent weights equal? {arrays_equal}") # True


agent.get_policy().model.base_model.load_weights("my_weights.h5") # now able to do inferences with rllib

del agent

ray.shutdown()

# for production without the need for ray as overhead:

import tensorflow as tf
import gym

class MyModel(tf.keras.Model):

    def __init__(self,num_actions):
      super().__init__()
      
      num_hidden = 256
      
      self.fc_1 = tf.keras.layers.Dense(num_hidden, activation="relu", name="fc_1")
      self.fc_value_1 = tf.keras.layers.Dense(num_hidden, activation="relu", name="fc_value_1")
      
      self.fc_2 = tf.keras.layers.Dense(num_hidden, activation="relu", name="fc_2")
      self.fc_value_2 = tf.keras.layers.Dense(num_hidden, activation="relu", name="fc_value_2")
      
      self.fc_out = tf.keras.layers.Dense(num_actions, activation="softmax", name="fc_out") # don't forget to include "softmax" here
      self.value_out = tf.keras.layers.Dense(1, name="value_out")
    
    def call(self, inputs):
        x_out = self.fc_1(inputs)
        x_value = self.fc_value_1(inputs)   
        
        x_out = self.fc_2(x_out)
        x_value = self.fc_value_2(x_value)  
        
        return [self.fc_out(x_out), self.value_out(x_value)]
        
env = gym.make("CartPole-v0")
model = MyModel(env.action_space.n) # pure tensorflow
obs = env.reset()
obs = tf.convert_to_tensor(obs)
obs = tf.expand_dims(obs, 0)
model.build(obs.shape)
model.load_weights("my_weights.h5")
model._name = "MyReplica" # replica of ray default model
model.summary()

"""
Model: "MyReplica"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 fc_1 (Dense)                multiple                  1280      
                                                                 
 fc_value_1 (Dense)          multiple                  1280      
                                                                 
 fc_2 (Dense)                multiple                  65792     
                                                                 
 fc_value_2 (Dense)          multiple                  65792     
                                                                 
 fc_out (Dense)              multiple                  514       
                                                                 
 value_out (Dense)           multiple                  257       
                                                                 
=================================================================
Total params: 134,915
Trainable params: 134,915
Non-trainable params: 0
_________________________________________________________________

"""

tf_weights = model.get_weights()

arrays_equal = all([np.array_equal(t,tfw) for t,tfw in zip(trained_weights, tf_weights)]) 

# => no changes as lr = 0.0 in last results.fit()
print(f"Trained and tensorflow weights equal? {arrays_equal}") # True

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    # env.render()
    obs = tf.convert_to_tensor(obs)
    obs = tf.expand_dims(obs, 0)
    action_probs, _ = model(obs)
    action = np.random.choice(len(np.squeeze(action_probs)), p=np.squeeze(action_probs))
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    
print(f"Tensorflow agent total reward: {episode_reward}") # 200
