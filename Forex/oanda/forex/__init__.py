import json
import os
import numpy as np

from .ForexApi import ForexApi 
from .utils import utils
from .utils import deriv, to_date, read_settings
from .hr import hr
# from .Forex import Forex 

from .min import min

# import database
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
    - _variable is to signify private variables
    - All pairs will be done as XXX_XXX, ex EUR_USD



"""


#### Hyperparameters ####

DATABASE_PATH = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.getcwd()


#########################


# Creation of the proper settings and metadata json.
# Metadata is for the entire class to use
# Settings is for the user to modify
_ = os.path.dirname(os.path.abspath(__file__))
_meta = os.path.join(_, 'metadata.json')
_settings = os.path.join(_, 'settings.json')

if not os.path.exists(_meta):
    with open(os.path.join(_, 'metadata.json'),'w+') as file:
        file.write("{}")
if not os.path.exists(_settings):
    with open(os.path.join(os.getcwd(),'settings.json'),'w+') as file:
        file.write("{}")




# writes all necessary data for metadata
with open(_meta, "r") as f:
    d = json.load(f)
d["DATABASE_PATH"] = DATABASE_PATH
d["SETTINGS_PATH"] = SETTINGS_PATH
with open(_meta,"w") as f:
    json.dump(d,f)




##### Used Functions ######


