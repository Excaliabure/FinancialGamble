import time 
import json
import forex as fx
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn


pairs =['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_NZD', 'AUD_SGD', 'AUD_USD', 
        'CAD_CHF', 'CAD_HKD', 'CAD_SGD', 'CHF_HKD', 'CHF_ZAR', 
        'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_CZK', 'EUR_DKK', 'EUR_GBP', 'EUR_HKD', 
        'EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_SGD', 
        'EUR_TRY', 'EUR_USD', 'EUR_ZAR', 'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 
        'GBP_NZD', 'GBP_PLN', 'GBP_SGD', 'GBP_USD', 'GBP_ZAR', 
        'NZD_CAD', 'NZD_CHF', 'NZD_HKD', 'NZD_SGD', 'NZD_USD', 'SGD_CHF',
        'USD_CAD', 'USD_CHF', 'USD_CNH', 'USD_CZK', 'USD_DKK', 
        'USD_HKD', 'USD_MXN', 'USD_NOK', 'USD_PLN', 'USD_SEK', 'USD_SGD', 
        'USD_THB', 'USD_TRY', 'USD_ZAR'
        # 'EUR_NOK', 'EUR_HUF',
        # 'ZAR_JPY','HKD_JPY', 'GBP_JPY', 'EUR_JPY', 'AUD_JPY', 'CAD_JPY', 'NZD_JPY', 'CHF_JPY', 'SGD_JPY', 'TRY_JPY', 'USD_JPY'
        
    ]




from oandapyV20 import API
apiKey = None
accountID = None
with open("dev_settings.json", "r") as file:
    d = json.load(file)
    apiKey = d["acckey"]
    accountID = d["accid"]


env = fx.ForexApi(apiKey, accountID)


TIMESTEP = 50 # In minutes, 0 < TIMESTEP < 51

prev = np.zeros((len(pairs),1))
curr = np.zeros((len(pairs),1))

env.close_all_orders(True)

for q in range(len(pairs)):
    
    data = env.get_pair(pairs[q])
        
    # CHLO 
    
    prev[q] = data[-TIMESTEP][0]
    curr[q] = data[-1][0]

import os
npyarr = None
if os.path.exists("monies.npy"):
    npyarr = np.load("monies.npy")
else:
    temp = [env.bal()['balance']]
    np.save("monies.npy", np.array(temp))
    npyarr = np.array(temp)
    
monies_arr = []

for q in npyarr:
    monies_arr.append(q)

for i in range(10):
    slope = (curr-prev) / curr * 100
    port = (np.zeros((slope.shape[0],1)) + 31000)


    decision = np.array(slope * port, dtype=int)

    for j in range(len(pairs)):
        env.buy_sell(pairs[j], decision[j].item(), 100)



    # time.sleep(TIMESTEP * 60 * 3.14)

    for i in range(60):
        monies_arr.append(env.bal()['NAV'])
        np.save('monies.npy',np.array(monies_arr))
        time.sleep(TIMESTEP * 3.14)

    env.close_all_orders(True)
    time.sleep(3)


    monies_arr.append(env.bal()['balance'])
    np.save('monies.npy',np.array(monies_arr))


    for q in range(len(pairs)):
    
        data = env.get_pair(pairs[q])
            
        prev[q] = data[-TIMESTEP][0]
        curr[q] = data[-1][0]

    
    