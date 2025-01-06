from oandapyV20 import API
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
import oandapyV20
import datetime
import random
import glob
import time
import json
import sys
import os

# import database
import utils
import sys

# from oandapyV20.contrib.requests import MarketOrderRequest
# from oandapyV20.endpoints.pricing import PricingStream
# import oandapyV20.endpoints.instruments as instruments
# from oandapyV20.endpoints.pricing import PricingInfo
# import oandapyV20.endpoints.positions as Positions
# import oandapyV20.endpoints.accounts as Account
# import oandapyV20.endpoints.pricing as Pricing

# import oandapyV20.endpoints.orders as Order
# from oandapyV20.exceptions import V20Error



"""
Attempt at making a class

Rules:

    - for libraries, make everything as basecase as 
    possible aka no plt.plot, yes matplotlib.pyplot.plot
    remove ambiguity
    - 

__init__ will:

    - create the database for the file 
    - managing access for the currency pairs 
    - updating if necessary


"""


#### Hyperparameters ####

DATABASE_PATH = '/'

#########################

# First metadata
# if not os.path.exists('metadata.json')):
#     with open('metadata.json','w+') as file:
#         file.write("{}")


# with open('metadata.json', 'w') as file:
#     f = json.load(file)
    # f['DATABASE_PATH'] = DATABASE_PATH




#### Creating Database for Files ####






# class test:
#     def __init__(self, val):
#         self._value = val

#     @property
#     def value(self):
#         return self._value
