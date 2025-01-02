from datetime import date, timedelta
import matplotlib.pyplot as plt
from oandapyV20 import API
from os.path import join
import yfinance as yf
import pandas as pd
import numpy as np
import oandapyV20
import datetime
import random
import glob
import time
import json
import sys
import os



from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.endpoints.pricing import PricingStream
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.endpoints.pricing import PricingInfo
import oandapyV20.endpoints.positions as Positions
import oandapyV20.endpoints.accounts as Account
import oandapyV20.endpoints.pricing as Pricing

import oandapyV20.endpoints.orders as Order
from oandapyV20.exceptions import V20Error



"""
Attempt at making a class


"""

class test:
    def __init__(self, val):
        self._value = val

    @property
    def value(self):
        return self._value