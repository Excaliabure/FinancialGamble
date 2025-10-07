import time 
import json
import forex as fx
import itertools
import numpy as np
from oandapyV20 import API
import matplotlib.pyplot as plt

pairs =['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_NZD', 'AUD_SGD', 'AUD_USD', 
        'CAD_CHF', 'CAD_HKD', 'CAD_SGD', 'CHF_HKD', 'CHF_ZAR', 
        'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_CZK', 'EUR_DKK', 'EUR_GBP', 'EUR_HKD', 
        'EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_SGD', 
        'EUR_TRY', 'EUR_USD', 'EUR_ZAR', 'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 
        'GBP_NZD', 'GBP_PLN', 'GBP_SGD', 'GBP_USD', 'GBP_ZAR', 
        'NZD_CAD', 'NZD_CHF', 'NZD_HKD', 'NZD_SGD', 'NZD_USD', 'SGD_CHF',
        'USD_CAD', 'USD_CHF', 'USD_CNH', 'USD_CZK', 'USD_DKK', 
        'USD_HKD', 'USD_MXN', 'USD_NOK', 'USD_PLN', 'USD_SEK', 'USD_SGD', 
        'USD_THB', 'USD_TRY', 'USD_ZAR',
        # 'EUR_NOK', 'EUR_HUF',
        # 'ZAR_JPY','HKD_JPY', 'GBP_JPY', 'EUR_JPY', 'AUD_JPY', 'CAD_JPY', 'NZD_JPY', 'CHF_JPY', 'SGD_JPY', 'TRY_JPY', 'USD_JPY'
        
    ]



apiKey = None
accountID = None
with open("dev_settings.json", "r") as file:
    d = json.load(file)
    apiKey = d["acckey"]
    accountID = d["accid"]


env = fx.ForexApi(apiKey, accountID)


TIMESTEP = 50 # In minutes, 0 < TIMESTEP < 51

arr = np.zeros 

print(env.get_pair("EUR_USD"))