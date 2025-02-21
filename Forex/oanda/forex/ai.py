
import gymnasium as gym
import numpy as np

"""
Structure of this module

Start with initializing it 
env = fx.ai.name_of_ai
env.init()

Load or start training a new model

env.load("name of model") 
or 
env.train("model name", config=None) # will use default params if config = None

Use env.list_models() to show all the current models


# input : y[0:64], buy[0:64], sell[0:64]
# output : none, buy, sell, close (Probabilities)
# Notes: uses reinforcement learning, thus buy and sell have 
# negative rewards and close has positive if positive pl or 
# negative if negative pl.

"""





class deriv12_env(gym.Env):
    def __init__(self, config=None):
     
        # import ray


        # ray.init(logging_level="info", log_to_driver=False)

        # config = ray.rllib.algorithms.ppo.PPOConfig()
        # config.framework("torch") 
        # # self.env = ForexApi(config["apiKey"], config["accountID"])

        self.observation_space = gym.spaces.Box(-1.0, 1.0, (64,), np.float32)
        self.action_space = gym.spaces.Discrete(3)

        # Actons : accept, descline, close

    def reset(self, seed=None, options=None):



        return np.array([1.0]), {}

    def step(self, action):
        # Return next observation, reward, terminated, truncated, and info dict.
        return np.array([1.0]), 1.0, True, False, {}

    
class deriv12:

    def __init__(self, config):
        self.cfg = config
        pass
    def build_and_run(self):
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        
        ray.shutdown()
        ray.init()
        self.ap = gym.spaces.Discrete(3)
        config = (
            PPOConfig()
            .framework("torch")
            .environment(
                deriv12,
                env_config={ "apiKey" : self.cfg["apiKey"],
                             "accountID" : self.cfg["accountID"]},  # `config` to pass to your env class
            )
            .debugging(log_level="ERROR")
            .env_runners(num_env_runners=0)
        )
        algo = config.build()
        print("Built")
    
        # print(algo.train())
        # checkpointDir = algo.save_to_path()
        # print(f"Saved checkpoint to {checkpointDir}")

