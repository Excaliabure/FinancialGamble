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
Rules

    - snake_case for functions 
    - camelCase for variables
    - SNAKE_CASE for hyperparameters
    - PascalCase for classes

    


"""
AllCurrenyPairs = [
    "EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", 
    "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/NZD",
    "GBP/JPY", "GBP/AUD", "GBP/CAD", "GBP/CHF", "GBP/NZD",
    "AUD/JPY", "AUD/CHF", "AUD/CAD", "AUD/NZD",
    "CAD/JPY", "CAD/CHF", "NZD/JPY", "NZD/CHF",
    "USD/SEK", "USD/NOK", "USD/DKK", "USD/ZAR", 
    "USD/TRY", "USD/MXN", "USD/HKD",
    "EUR/SEK", "EUR/NOK", "EUR/DKK", "EUR/ZAR", 
    "EUR/TRY", "EUR/MXN", "EUR/HKD",
    "GBP/SEK", "GBP/NOK", "GBP/DKK", "GBP/ZAR", 
    "GBP/TRY", "GBP/MXN", "GBP/HKD"
]

class Forex:
    def __init__(self):
        print("Working")
    pass


class ForexApi:

    pass