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

##### Getters

def get_min(pair, all_days=False):
    """ Gets the most recent min data """
    pair = pair+"=X"  
    if (all_days):
        t = datetime.datetime.today() - timedelta(days=29)
        initial = yf.download(pair, interval= "1m", start = t, end = t + timedelta(days=5) ,progress=False, ignore_tz=True).reset_index()
        t += timedelta(days=5)

        while (t < datetime.datetime.today()):
            i2 = yf.download(pair, interval= "1m", start = t, end = t + timedelta(days=5) ,progress=False, ignore_tz=True).reset_index()
            t += timedelta(days=5)
            initial = pd.concat([initial,i2], axis = 0)

        initial.drop_duplicates(inplace=True)
        initial.Datetime = initial.Datetime.apply(lambda x: x.timestamp())
        return initial

    else:
        data = yf.download(pair, interval= "1m", period = "max",progress=False, ignore_tz=True).reset_index()
    data["Datetime"] = data.Datetime.apply(lambda x: x.timestamp())
    return data.iloc[:,0:6]


def get_hr(pair):
    """ Gets the most recent hr data from yf"""
    pair = pair.replace("_","")
    pair = pair+"=X"
    data = yf.download(pair, interval= "1h", period = "2y",progress=False, ignore_tz=True).reset_index()
    data["Datetime"] = data.Datetime.apply(lambda x: x.timestamp())
    return data.iloc[:,0:6]

def get_day(pair):
    """ Gets the most recent day data from yf"""
    pair = pair+"=X"
    data = yf.download(pair, interval= "1d", period = "max",progress=False, ignore_tz=True).reset_index()
    data.rename(columns={"Date":"Datetime"},inplace=True)
    data["Datetime"]  = data.Datetime.apply(lambda x: x.timestamp())
    return data.iloc[:,0:6]

def _json_save(file,key,data):
    
    

    with open(file, 'r') as file:
        data = json.load(file)

  
    data[key] = data
    with open(file, 'w') as file:
        json.dump(data, file, indent=4) 



##### Utility


