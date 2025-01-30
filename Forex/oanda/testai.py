
from ray.rllib.algorithms.ppo import PPOConfig


config = (
    PPOConfig()
    .framework("torch")
    .environment("CartPole-v1",)
    .rollouts(num_rollout_workers=0)  # Optional: use only the local worker
    # .debugging(log_level="ERROR")  # Set log level to ERROR to suppress most messages
)

algo = config.build()
print(algo.train())