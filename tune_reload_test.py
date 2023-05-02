#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:43:12 2023

@author: Jorgen Svane
"""

import os
import numpy as np
import ray
from ray import air, tune
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()

config = PPOConfig()\
        .training(lr=tune.grid_search([0.01, 0.001, 0.0001]))\
        .rollouts(num_rollout_workers=4)\
        .framework(framework="tf2",eager_tracing=True)\
        .resources(num_gpus=0)\
        .environment(env="CartPole-v1")
    
# fist tuning to actually get a trained algo
tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"training_iteration": 100,
                                    "episode_reward_mean": 500},
                              local_dir="ppo_test"),
    param_space=config
    )

tuner.fit()

# ray.shutdown()

# reload the previous tuner
restored_tuner = tune.Tuner.restore(path="/home/novelty/ray240_tune_ppo_tf_test/ppo_test/PPO",trainable="PPO")
result_grid = restored_tuner.get_results()

# Check if there have been errors
if result_grid.errors:
    print("One of the trials failed!")
else:
    print("No errors!")
    
num_results = len(result_grid)
print("Number of results:", num_results)

# Iterate over results
for i, result in enumerate(result_grid):
    if result.error:
        print(f"Trial #{i} had an error:", result.error)
        continue

    print(
        f"Trial #{i} finished successfully with a mean accuracy metric of:",
        result.metrics["episode_reward_mean"]
    )
    
results_df = result_grid.get_dataframe()
results_df[["training_iteration", "episode_reward_mean"]]

print("Shortest training time:", results_df["time_total_s"].min())
print("Longest training time:", results_df["time_total_s"].max())

best_result_df = result_grid.get_dataframe(
    filter_metric="episode_reward_mean", filter_mode="max"
)
best_result_df[["training_iteration", "episode_reward_mean"]]

from ray.air import Result

# Get the result with the maximum test set `episode_reward_mean`
best_result: Result = result_grid.get_best_result(metric="episode_reward_mean", mode="max")

# Get the result with the minimum `episode_reward_mean`
worst_performing_result: Result = result_grid.get_best_result(
    metric="episode_reward_mean", mode="min"
)

best_result.config
best_result.config["lr"]

best_result.log_dir

# Get the last reported set of metrics
best_result.metrics

result_df = best_result.metrics_dataframe
result_df[["training_iteration", "episode_reward_mean", "time_total_s"]]

"""
algo = PPO(config=config, env= "CartPole-v1")
Out:
2023-05-02 11:29:26,131	WARNING deprecation.py:50 -- DeprecationWarning: `algo = Algorithm(env='CartPole-v1', ...)` 
has been deprecated. Use `algo = AlgorithmConfig().environment('CartPole-v1').build()` instead. 
This will raise an error in the future!
"""
# right way now
algo = (
        PPOConfig()
        .training(lr=best_result.config["lr"])
        .rollouts(num_rollout_workers=4)
        .environment('CartPole-v1')
        .framework(framework="tf2",eager_tracing=True)
        .resources(num_gpus=1)
        .build()
        )
algo.restore(best_result.checkpoint)

policy = algo.get_policy()
fcn = policy.model # instance of ray.rllib.models.tf.fcnet.FullyConnectedNetwork
fcn.base_model._name = "MyDefaultFcn" # give the model a more descriptive name
fcn.base_model.summary()

"""
Out:
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

# algo.get_policy().model.base_model.load_weights("my_weights.h5")

# fcn.base_model.save("my_model.h5")

trained_weights = algo.get_policy().get_weights()

new_algo =  (
            PPOConfig()
            .training(lr=best_result.config["lr"])
            .environment('CartPole-v1')
            .framework(framework="tf2",eager_tracing=True)
            .build()
            )

untrained_weights = new_algo.get_policy().get_weights()

# check whether weights are equal

arrays_equal = all([np.array_equal(t,u) for t,u in zip(trained_weights, untrained_weights)])

print(f"Trained and untrained weights equal? {arrays_equal}") # False

new_algo.get_policy().model.base_model.load_weights("my_weights.h5")

loaded_weights = new_algo.get_policy().get_weights()

arrays_equal = all([np.array_equal(t,l) for t,l in zip(trained_weights, loaded_weights)])

print(f"Trained and loaded weights equal? {arrays_equal}") # True

del algo

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
    
config = PPOConfig()\
        .training(lr=0.0)\
        .rollouts(num_rollout_workers=4)\
        .framework(framework="tf2",eager_tracing=True)\
        .resources(num_gpus=0)\
        .environment(env="CartPole-v1")
        
# second tuning with lr = 0.0 to confirm weights are loaded and not changed
# hence, config changed relative to first tuning
tuner = tune.Tuner(
    PPOalgo,
    run_config=air.RunConfig(name="PPOalgo",
                            stop={"training_iteration": 1, # note only one iteration
                                    "episode_reward_mean": 450},
                            local_dir="ppo_test"),
    param_space=config
    )

tuner.fit()

restored_tuner = tune.Tuner.restore(path="/home/novelty/ray240_tune_ppo_tf_test/ppo_test/PPOalgo",trainable=PPOalgo)
result_grid = restored_tuner.get_results()
best_result: Result = result_grid.get_best_result(metric="episode_reward_mean", mode="max")

print("Lets make sure we pull the right checkpoint\n and not the one created by\n the original PPO:\n", best_result.checkpoint)

"""
Out:
Checkpoint(local_path=/home/novelty/ray240_tune_ppo_tf_test/ppo_test/PPOalgo/PPOalgo_CartPole-v1_ac5b9_00000_0_2023-05-02_12-39-53/checkpoint_000001)
"""

# We deleted the old algo so let's use new_algo and bring it up to speed
new_algo.restore(best_result.checkpoint)
# or do this
# new_algo.get_policy().model.base_model.load_weights("my_weights.h5") # now able to do inferences with rllib

# Now able to continue training in rllib if needed - remember to reset lr to something != 0.0 !!!
# Rebuild the new_algo again with new lr - see above
# For now we can do inferences with rllib

weights = new_algo.get_policy().get_weights()

arrays_equal = all([np.array_equal(t,w) for t,w in zip(trained_weights, weights)]) # True => no changes as lr = 0.0

print(f"Trained and recent weights equal? {arrays_equal}") # True

del new_algo

ray.shutdown()

# for production without the need for ray as overhead:

import tensorflow as tf
import gymnasium as gym

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
        
env = gym.make("CartPole-v1")
# env = gym.make("CartPole-v1", render_mode="human")
model = MyModel(env.action_space.n) # pure tensorflow
obs,_ = env.reset()
obs = tf.convert_to_tensor(obs)
obs = tf.expand_dims(obs, 0)
model.build(obs.shape)
model.load_weights("my_weights.h5")
model._name = "MyReplica" # replica of ray default model
model.summary()

"""
Out:
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

# => no changes as lr = 0.0 in last tuner.fit()
print(f"Trained and tensorflow weights equal? {arrays_equal}") # True

# run until episode ends
episode_reward = 0
terminated, truncated = False, False
obs,_ = env.reset()
while not (terminated or truncated):
    obs = tf.convert_to_tensor(obs)
    obs = tf.expand_dims(obs, 0)
    action_probs, _ = model(obs)
    action = np.random.choice(len(np.squeeze(action_probs)), p=np.squeeze(action_probs))
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
env.close()
    
print(f"Tensorflow agent total reward: {episode_reward}") # app. 500