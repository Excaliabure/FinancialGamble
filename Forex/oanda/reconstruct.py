import matplotlib.pyplot as plt
import forex as fx
import json
import numpy as np

apiKey = None
accountID = None
with open("dev_settings.json", "r") as file:
    d = json.load(file)
    accountID = d["acckey"]
    apiKey = d["accid"]



a = fx.min("EURCHF")#.to_numpy()[0][:,2]
print(a)






# env = fx.ForexApi(apiKey,accountID)
# data = fx.hr("EUR_USD")

# y = data.to_numpy()[0]

# import ray
# from ray.rllib.algorithms.ppo import PPOConfig
# config = (
#     PPOConfig()
#     .framework("torch")
#     .environment(
#         fx.ai.deriv12_env,
#         env_config={
#                 "accountID" : accountID,
#                 "apiKey" : apiKey
#             },  # `config` to pass to your env class
#     )
#     .debugging(log_level="ERROR")
#     .env_runners(num_env_runners=0)
# )
# algo = config.build()

# g = fx.ai.deriv12({
#     "accountID" : accountID,
#     "apiKey" : apiKey
# })

# g.build_and_run()


# print(f"Descrete Value : {g.ap}")


"""
Desired
import forex as fx
y = fx.min("Eur USD")
y.info()

Columns be (...,...,...,...,...,...) 

oh no datetimes are jsut big numbers

fx.to_date(timestamp)



"""
