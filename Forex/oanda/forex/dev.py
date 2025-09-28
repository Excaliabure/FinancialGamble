
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


def database_save(data):
    
    base_dir = os.path.join(os.path.dirname(__file__), "database")

    current_pairs = glob.glob(base_dir)