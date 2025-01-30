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

from ._utils import _json_save

APIKEY = None
ACCOUNTID = None


class ForexApi():

    def __init__(self, apikey=None, accountid=None):
        # Preprocessing login info
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


        # Adding creating environment to allow for trading
        self.api = API(access_token=apikey, environment="practice", headers={"Accept-Datetime-Format": "UNIX"})
        
    

    def buy_sell(self, pair, units, pip_diff, view=False, terminal_print=True, time_In_Force="FOK",type_="MARKET", price="1"):
        """ 
        :params
            pair - forex pair, ex [EURUSD EUR/USD EUR_USD] are all valid formats
            units - How much to buy. - value makes sell postiion and + makes but position
            view - Doesnt execute the order, just displays the order to fill
        If position is negative, sell pos, else pos buy pos"""

        
        p = (str(units) if type(units) == str else units)
        pip = 1e-4
        p = pair

        if "_" not in p:
            p = pair[:3] + "_" + pair[3:]

        request = PricingInfo(accountID=self.accountID, params={"instruments": p})
        
        response = self.api.request(request)

        # Extract bid and ask prices from the response
        prices = response.get("prices", [])
        asset_price = float(prices[0]['asks'][0]['price'])
        basediff = pip * pip_diff
        tp = asset_price + (-basediff if units  < 0 else basediff)
        sl = asset_price - (-basediff if units  < 0 else basediff)

        order_info = {
            "order": {
                "price": price,
                "takeProfitOnFill": {
                        "timeInForce": "GTC",
                        "price": str(round(tp,5))
                    },
                "stopLossOnFill": {
                    "timeInForce": "GTC",
                    "price": str(round(sl,5))
                    },
                
            "timeInForce": "FOK",
            "instrument": p,
            "units": str(units),
            "type": type_,
            "positionFill": "DEFAULT"
            }
        }
        

        if terminal_print:
            print(order_info)
        
        if view:
            return order_info
        else:
            o = Order.OrderCreate(self.accountID,order_info)
            resp = self.api.request(o)
            
            return resp
    

    def get_pair(self, _pair, granularity="M1", return_price_1=True):
        """
        M1 is for minute 1
        H1 is for hour 1
        D1 is for day 1
        
        """
        p = _pair
        if "_" not in p:
            p = _pair[:3] + "_" + _pair[3:]


        current_time = datetime.datetime.now()
        start_time = (current_time - datetime.timedelta(minutes=50)).isoformat() + "Z"
        end_time = current_time.isoformat() + "Z"

        parm = {
            "instruments" : p,
            "granularity": granularity,
            "from": start_time,
            "to": end_time
        }

        
        request = PricingInfo(accountID=self.accountID, params=parm)
        response = self.api.request(request)

        # Extract bid and ask prices from the response
        prices = response.get("prices", [])
        asset_price = float(prices[0]['asks'][0]['price'])

        return response


    def close(self, _pair):
        """ Closes specific order"""
        

        pair = (_pair if "_" in _pair else _pair[:3] + "_" + _pair[3:])
        list_orders = Positions.OpenPositions(self.accountID)
        order_dict = self.api.request(list_orders)
        plist = order_dict['positions']
        pair_info = None

        for i in plist:
            if i['instrument'] == pair:       
                pair_info = plist[0]
            else:
                pair_info = None    

        if pair_info == None:
            return -1
        else:
            toclose = ({"longUnits" : "ALL"} if int(pair_info['long']['units']) != 0 else {"shortUnits" : "ALL"})
        
        try:
            req = Positions.PositionClose(accountID=self.accountID, instrument=pair, data=toclose)
            respo = self.api.request(req)
            return respo
        except:
            return 0
        
    def view(self,_pair=None,gen_info=False):
        """ Views info of pair """

        list_orders = Positions.OpenPositions(self.accountID)
        account_info = Account.AccountDetails(self.accountID)
        positions = self.api.request(list_orders)
        acc_info = self.api.request(account_info)
        
        if gen_info:
            return acc_info

        if _pair == None:
            return positions

        else:
            pair = (_pair if "_" in _pair else _pair[:3] + "_" + _pair[3:]) 

            for i in positions['positions']:
                if i['instrument'] == pair:
                    return i
        # Return None if not found
        return None