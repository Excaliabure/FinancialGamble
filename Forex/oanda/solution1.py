import datetime 
import random
import time
# from utils import log, smooth_ma
import json
import os
import numpy as np
import sys
import forex as fx

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


######
# Modify the values 
APIKEY = None
ACCOUNTID = "101-001-27337634-002"



######



SETTINGS = {
    "Settings": {
        "Api Key": APIKEY,
        "Account ID": ACCOUNTID,
        "Trade Duration": 28800,
        "Trade Interval": 60,
        "Iterations" : 2000,
        "coef" : 0.5,
        "General Settings" : "true",
        "units" : 1000,
        "sltp" : 30,
        "count" : 7,
        "tolerance": 0.0001, 


        "Pair Settings": {
        
            "AUD_CAD": {
                "units": 1000,
                "sltp": 1000,
                "count": 2
            }
        
        }        
    }
}



    

def deriv(arr):
    """ Descrete derivative """
    
    darr = np.zeros(len(arr))

    for i in range(1,len(darr)):
        darr[i] = arr[i] - arr[i-1]
    
    if len(darr) > 2:
        darr[-1] = darr[-2] - darr[-3]

    if len(darr) <= 1:
        return np.array([0])
    
    darr[0] = darr[1]
    
    return darr


def smooth_ma(arr_, amt=6):
    """ Smooths out hte curve for better derivatives
    Not perfect, but get good enough smoothing"""
    arr = arr_.copy()
    arr[0] = arr_[0]
    for j in range(amt):
        for i in range(2,len(arr)):
            mid = (arr[i] - arr[i-1])/2
            arr[i-1] += mid
            arr[i] -= mid
    return arr


def start(dict=None,log_off=False):
    if dict == None:
        settings = json.load(open("settings.json"))["Settings"]
    else:
        settings=dict
    required = ["Api Key","Account ID","Trade Duration", "Trade Interval"]
    # Makes sure settings are proper
    # missing = []
    # for i in required:
    #     if i not in settings:
    #         missing.append(i)
    # if len(missing) != 0:
    #     print("Missing parameters in settings: \n")
    #     for i in missing:
    #         print(f"\t-{i}\n")
    # if len(missing) != 0:
    #     sys.exit(0)
    start_time = datetime.datetime.now().timestamp()
    s = f"Started at {start_time:.4f}\n"
    print(s)
    
    return start_time, settings


def data_arr_collection(filename,name,val, overwrite = False):

    # if '.json' not in file:
    #     file = file + '.json'

    with open(filename, 'r') as file:
        data = json.load(file)
        # print("Data read from file:", data)

    if name not in data.keys():
        data[name] = []
    
    data[name].append(val)
    if overwrite:
        data[name] = val
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4) 
    return data


if __name__ == '__main__':
    import forex as fx
    apiKey = None
    accountID = None
    with open("dev_settings.json", "r") as file:
        d = json.load(file)
        apiKey = d["acckey"]
        accountID = d["accid"]
    APIKEY = apiKey
    ACCOUNTID = accountID
    SETTINGS["Api Key"] =  APIKEY
    SETTINGS["Account ID"] = ACCOUNTID

    if APIKEY != None:
        ap = APIKEY
    else:
        ap = input("Consider Edititng the file\nInput Api Key: ")
    
    if ACCOUNTID != None:
        ai = ACCOUNTID
    else:
        ai = input("Consider Editing the file\nInput Account id: ")

    env = fx.ForexApi(APIKEY, ACCOUNTID)
    env.log_info(log_off=True)
    start_time, settings = start(dict=SETTINGS, log_off=True)

    if not os.path.exists('data.json'):
        with open('data.json','w+') as file:
            file.write("{}")
            file.close()
    print("\n")
    
    ############ Code goes here ##################

    arrPnL = []
    pullout = False
    prevDecision = 0
    c = 0
    # print(env.view("EUR_USD"))
    while True:
        minute = fx.min("EUR_USD").to_numpy()
        hr = fx.hr("EUR_USD").to_numpy()


        if len(minute) > 0  and len(hr) > 0:
            minute = minute[0][:,2]
            hr = hr[0][:,2]
            
            dm = fx.algo.deriv12(minute)
            dh = fx.algo.deriv12(hr)
            


            if dm == dh  and prevDecision != dh and pullout:
                print(f"{'Bought' if dh < 0 else 'Sold'}")
                env.close("EUR_USD")
                time.sleep(2)
                if prevDecision != 0:
                    env.buy_sell("EUR_USD",-1000 * dh, 300)
                
                prevDecision = dh
                data_arr_collection('data.json', 'bal', float(env.view(gen_info=True)['account']['balance']))


            while (env.view("EUR_USD") == None and prevDecision != 0 and pullout == False):
                print()
                print(f"Non sell/buy position for EUR_USD. Attempting...")
                time.sleep(0.5)
                env.buy_sell("EUR_USD", -1000 * prevDecision, 300, terminal_print=False)
                print()
                time.sleep(0.5)

            

            time.sleep(30)
            minute = fx.min("EUR_USD").to_numpy()[0][:,2]
            hr = fx.hr("EUR_USD").to_numpy()[0][:,2]
            day = fx.day("EUR_USD").to_numpy()[0][:,2]


            if env.view("EUR_USD") != None:
                arrPnL.append(float(env.view("EUR_USD")["unrealizedPL"])    )
                if fx.algo.deriv12(arrPnL) == -1:
                    print(f"Exited with PL of {env.view('EUR_USD')['unrealizedPL']}")
                    pullout = True
                    env.close("EUR_USD")
                    data_arr_collection("data.json",f"pl_{c}", float(env.view("EUR_USD")["unrealizedPL"]))
                    c += 1

                
        


    # d = data_arr_collection("data.json", "PnL", env.)