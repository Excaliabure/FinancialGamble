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


"""
Will be used for databasing currency pairs for ananlysis

    - Create folder and verify folder 

"""


def init(path):

    # Check if path to folder exists
    if os.path.exists(path):
        return
    else:
        

