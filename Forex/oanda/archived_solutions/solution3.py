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
from tqdm import tqdm

ENV_CONFIG = {"pair" : "EUR_USD",
              "time" : "hour"}

class sol3_env(gym.Env):
    def __init__(self, config=None):


        # self.history = fx.min("EUR_USD").to_numpy()[0][:,1:5]
        # if "pair" not in config.keys():
        if config["time"] in ["hour", "hr", "H"]:
            self.history = fx.hr(config["pair"]).to_numpy()[0][:,1:5]
        elif config["time"] in ["minute", "min", "m"]:
            self.history = fx.min(config["pair"]).to_numpy()[0][:,1:5]
        else:
            self.history = fx.day(config["pair"]).to_numpy()[0][:,1:5]
        
        self.hr = fx.hr(config["pair"]).to_numpy()[0][:,1:5]
        self.min = fx.min(config["pair"]).to_numpy()[0][:,1:5]
        self.day = fx.day(config["pair"]).to_numpy()[0][:,1:5]
        # self.history = np.vstack((self.day, self.hr,self.min))/
        self.history = self.hr
            
            
            
        self.action_space = gym.spaces.Discrete(3) 
        r1 = random.randint(0,len(self.history) - 128)
        _state = self.history[r1: r1+64]
        self._s1 = _state[:32]#.flatten()
        self._s2 = _state[32:]#.flatten()
        print(f"SHAPE #######################\n{self._s1.shape}")
        self.c = 0
        self.iters = 50000
        self.total_reward = 0
        self.observation_space = gym.spaces.Box(0.0, 2.0, shape=(32,4), dtype=np.float64)
        

        ### test
        self.monie = 100
        

    def reset(self, seed=None, options=None):

        r1 = random.randint(0,len(self.history) - 128)
        _state = self.history[r1: r1+64]
        self._s1 = _state[:32]#.flatten()
        self._s2 = _state[32:]#.flatten()
        # print(f"SHAPE #########################\n{self._s1.shape} {self._s2.shape}")
        self.counting = 0
        self.monie=100

        # return np.array([1.0]), {}
        return self._s1, {"total_reward" : self.total_reward}

    def step(self, action):
        
        reward = 0
        terminated = False
        
        s1_v = self._s1[-1][1]
        s2_v = self._s2[-1][1]
        # print(self._s1[-1])

        if action == 0:
            # buy 
            # print("[BUY]")
            if s1_v < s2_v:
                reward = s2_v - s1_v
                reward = 1
            else:
                reward = -(s2_v - s1_v)
                reward = -1  
        if action == 1:
            # sell
            # print("[SELL]")
            if s1_v > s2_v:
                reward = s2_v - s1_v
                reward = 1
            else:
                reward = -(s2_v - s1_v)
                reward = -1
        
        if action == 2:
            # hold
            # print("[HOLD]")
            reward = 0
            
        
        r1 = random.randint(0,len(self.history) - 128)
        _state = self.history[r1: r1+64]
        self._s1 = _state[:32]#.flatten()
        self._s2 = _state[32:]#.flatten()

        # hist_stats = {
        # "total_reward": self.total_reward,
        # "average_reward": self.total_reward / (self.counting if self.counting > 0 else 1),
        # "episode_length": self.counting
        # }

        self.monie += reward
        # print(self.monie)
        if reward < -0.5:
            return self._s1, reward, True, False, {}#{"total_reward": reward, "hist_stats": hist_stats}

        self.counting += 1
        self.total_reward += reward
        
        # Return next observation, reward, terminated, truncated, and info dict (now with hist_stats)
        return self._s1, reward, False, False, {}#{"custom_metrics" : "Hi"}#{"total_reward": reward, "hist_stats": hist_stats}



def env_creator(env_config):
    return sol3_env(ENV_CONFIG)

def train_new(episodes = 1, config=None):
    ray.shutdown()
    ray.init()

    register_env("sol3", env_creator)
    
    _gamma=0.05 
    _lr=0.0001
    _train_batch_size_per_learner=1024
    _entropy_coeff=0.9
    _model={"fcnet_hiddens": [64, 64, 64, 64], "fcnet_activation": "relu"}  # Example NN

    config = PPOConfig()

    config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)

    config.environment("sol3")
    config.env_runners(num_env_runners=1)
    config.training(
        gamma=_gamma, 
        lr=_lr,
        lambda_=1,
        train_batch_size_per_learner=_train_batch_size_per_learner,
        entropy_coeff=_entropy_coeff,
        model=_model
        

    )
    # config.training(

    #     gamma=0.9, 
    #     lr=0.0001,
    #     train_batch_size_per_learner=256,
    #     entropy_coeff=0.9,
    #     model={"fcnet_hiddens": [128, 64], "fcnet_activation": "relu"}  # Example NN
    # )
    config.framework(framework="torch")

    # Build a Algorithm object from the config and run 1 training iteration.
    algo = config.build_algo()

    EPISODES = episodes
    reward_arr = []

    print("\n\n #################### TRAINING ################ \n\n")

    res = algo.train()
        
    for e in tqdm(range(EPISODES)):
        res = algo.train()
        reward_arr.append(res["env_runners"]["episode_reward_mean"])
    
    print(res)
    checkpoint_path = r"models"
    
    assert os.path.exists("models"), "models folder does not exist"

        
    algo.save(checkpoint_path)
    ray.shutdown()
    return reward_arr


def train_existing(episodes = 1):
    ray.shutdown()
    ray.init()

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

    EPISODES = episodes

    reward_arr = []

    res = algo.train()

    for e in tqdm(range(EPISODES)):
        res = algo.train()
        

    assert os.path.exists("models"), "models folder does not exist"
    algo.save(checkpoint_path)
    
    ray.shutdown()
    
def infer():
    register_env("sol3", env_creator)
    
    config = PPOConfig()
    config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)

    config.environment("sol3")
    config.env_runners(num_env_runners=1)
    config.training(
        gamma=0.9, lr=0.01, train_batch_size_per_learner=256
    )

    checkpoint_path = os.path.join(os.getcwd(),"models")
    trainer_restored = ray.rllib.algorithms.ppo.PPO.from_checkpoint(checkpoint_path)

    policy = trainer_restored.get_policy()
    data = fx.min("EUR_USD").to_numpy()[0][:,1:5][-64:-32]
    


    return policy.compute_actions(data)
    


def main():
    usr = None
    running = True

    while running:
        usr = input(f"{ENV_CONFIG['pair']} - {ENV_CONFIG['time']}\n[1] Train New\n[2] Train Existing\n[3] Inference Most Recent\n[4] Change Pair\nInput : ")


        if usr == "1":
            eps = int(input("Episodes: "))
            r = train_new(eps)
            plt.plot(r)
            plt.show()
        elif usr == "2":
            eps = int(input("Episodes: "))
            train_existing(eps)
            pass
        elif usr == "3":
            inf_data = infer()
            os.system("cls")
            print(inf_data)
            if inf_data[0][0] == 0:
                print(f"-- [BUY] -- ")
            elif inf_data[0][0] == 1:
                print("-- [SELL] --")
            else:
                print("-- [HOLD] --")
        elif usr == "4":
            pair = input("0 for no change\nPair : ")
            time = input("Time : ")
            if pair != "0":
                ENV_CONFIG["pair"] = pair 
            if time != "0":
                ENV_CONFIG["time"] = time   
            try:
                os.system("clear")
                os.system("cls")
            except:
                os.system("cls")
        else:
            running = False


# def debug():
#     t = sol3_env(ENV_CONFIG)

#     t.step(1)

main()
# debug()

