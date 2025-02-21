import matplotlib.pyplot as plt
import forex as fx
import json
import numpy as np
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import random
from ray.tune.registry import register_env
import os

class sol3_env(gym.Env):
    def __init__(self, config=None):
     
        self.history = fx.min("EUR_USD").to_numpy()[0][:,2]
        self.observation_space = gym.spaces.Box(0.0, 2.0, (32,), np.float64)
        self.action_space = gym.spaces.Discrete(2) 
        r1 = random.randint(0,len(self.history) - 128)
        _state = self.history[r1: r1+64]
        self._s1 = _state[:32]
        self._s2 = _state[32:]
        self.c = 0
        self.iters = 5000
        self.total_reward = 0
        

    def reset(self, seed=None, options=None):
        r1 = random.randint(0,len(self.history) - 128)
        _state = self.history[r1: r1+64]
        self._s1 = _state[:32]
        self._s2 = _state[32:]
        self.counting = 0

        # return np.array([1.0]), {}
        return self._s1, {"total_reward" : self.total_reward}

    def step(self, action):
        reward = 0
        terminated = False
        

        if action == 0:
            # buy 
            if self._s1[-1] < self._s2[-1]:
                reward = self._s2[-1] - self._s1[-1]
            else:
                reward = -(self._s2[-1] - self._s1[-1])
        if action == 1:
            
            if self._s1[-1] > self._s2[-1]:
                reward = self._s2[-1] - self._s1[-1]
            else:
                reward = -(self._s2[-1] - self._s1[-1])
            
        
        r1 = random.randint(0,len(self.history) - 128)
        _state = self.history[r1: r1+64]
        self._s1 = _state[:32]
        self._s2 = _state[32:]

        if self.counting > self.iters:
            return self._s1, reward, False, False, {"total_reward" : reward}

        self.counting += 1
        self.total_reward += reward
        # Return next observation, reward, terminated, truncated, and info dict.
        return self._s1, reward, False, False, {"total_reward" : reward}



def env_creator(env_config):
    return sol3_env(env_config)

def train_new():
    ray.init()
    register_env("sol3", env_creator)

    config = PPOConfig()

    config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)

    config.environment("sol3")
    config.env_runners(num_env_runners=1)
    config.training(
        gamma=0.9, lr=0.01, train_batch_size_per_learner=256
    )

    # Build a Algorithm object from the config and run 1 training iteration.
    algo = config.build_algo()

    EPISODES = 500

    reward_arr = []

    print("\n\n #################### TRAINING ################ \n\n")
    res = algo.train()
    
    for e in range(EPISODES):
        res = algo.train()

    checkpoint_path = r"models"
    
    assert os.path.exists("models"), "models folder does not exist"

        
    algo.save(checkpoint_path)


def train_existing():
    register_env("sol3", env_creator)

    checkpoint_path = os.path.join(os.getcwd(),"models")
    algo = ray.rllib.algorithms.ppo.PPO.from_checkpoint(checkpoint_path)
    print("Model restored from checkpoint.")
    
    config = PPOConfig()

    config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)

    config.environment("sol3")
    config.env_runners(num_env_runners=1)
    config.training(
        gamma=0.9, lr=0.01, train_batch_size_per_learner=256
    )

    # Build a Algorithm object from the config and run 1 training iteration.
    algo = config.build_algo()

    EPISODES = 5

    reward_arr = []

    res = algo.train()

    for e in range(EPISODES):
        res = algo.train()

    assert os.path.exists("models"), "models folder does not exist"
    algo.save(checkpoint_path)
    
    
def infer():

    checkpoint_path = os.path.join(os.getcwd(),"models")
    trainer_restored = ray.rllib.algorithms.ppo.PPO.from_checkpoint(checkpoint_path)

    policy = trainer_restored.get_policy()
    data = fx.min("EUR_USD").to_numpy()[0][:,2][-32:]
    return policy.compute_actions(data)
    pass


def main():
    usr = None
    running = True
    
    while running:
        usr = input("*Note these are in EURUSD\Minutes*\n[1] Train New\n[2] Train Existing\n[3] Inference Most Recent\nInput : ")

        if usr == "1":
            train_new()
        elif usr == "2":
            train_existing()
            pass
        elif usr == "3":
            print(infer())
            
        else:
            running = False



    pass


main()