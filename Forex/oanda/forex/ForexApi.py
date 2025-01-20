import matplotlib.pyplot as plt
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

# from newsapi import NewsApiClient

from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.endpoints.pricing import PricingStream
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.endpoints.pricing import PricingInfo
import oandapyV20.endpoints.positions as Positions
import oandapyV20.endpoints.accounts as Account
import oandapyV20.endpoints.pricing as Pricing

import oandapyV20.endpoints.orders as Order
from oandapyV20.exceptions import V20Error
from oandapyV20 import API

from .utilities._utils import _json_save

APIKEY = None
ACCOUNTID = None


class ForexApi():

    def __init__(self, apikey=None, accountid=None):
        self.apiKey = None
        self.accountID = None
        if apikey != None:
            self.apiKey = apikey
        if accountid != None:
            self.accountID = accountid

        _p = 0
        if (APIKEY == None and self.apiKey == None):
            usrKey = input("Enter Api Key : ")
            _p += 1
        if (ACCOUNTID == None and self.accountID == None):
            usrID = input("Enter Account ID : ")
            _p += 1
        
        if (_p > 0):
            usr = input("Would you like to save? [y/N]")
            if usr in ["y","Y","yes","Yes"]:
                with open("metadata.json", "r") as f:
                    d = json.load(f)
                
                d["ACCOUNTID"] = usrID
                d["APIKEY"] = usrKey
                with open("metadata.json","w") as f:
                    json.dump(d,f)

    
    

        
    def info():
        pass


    


