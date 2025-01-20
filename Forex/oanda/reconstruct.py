import forex
import numpy as np
import os
import yfinance as yf
import json

import forex.utilities

with open("dev_settings.json","r") as file:
    devSettings = json.load(file)

os.system('cls')

y = forex.ForexApi(apikey=devSettings["acckey"], accountid=devSettings["accid"])

print(y)

"""
Desired
import forex as fx
y = fx.min("Eur USD")
y.info()

Columns be (...,...,...,...,...,...) 

oh no datetimes are jsut big numbers

fx.to_date(timestamp)



"""
