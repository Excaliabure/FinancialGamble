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
# Import the specific exception to catch OANDA API errors
from oandapyV20.exceptions import V20Error
from oandapyV20 import API

# from ._utils import _json_save # Commented out as _utils is not provided

APIKEY = None
ACCOUNTID = None

class ForexApi():

    def __init__(self, apikey=None, accountid=None):
        # Preprocessing login info
        self.apiKey = apikey
        self.accountID = accountid

        # Logic for getting/saving API key and Account ID (kept mostly as is)
        _p = 0
        usrKey = None
        usrID = None

        if (APIKEY == None and self.apiKey == None):
            usrKey = input("Enter Api Key : ")
            self.apiKey = usrKey
            _p += 1
        if (ACCOUNTID == None and self.accountID == None):
            usrID = input("Enter Account ID : ")
            self.accountID = usrID
            _p += 1
        
        if (_p > 0):
            usr = input("Would you like to save? [y/N]")
            if usr in ["y","Y","yes","Yes"]:
                try:
                    # Check if metadata.json exists and read it
                    if os.path.exists("metadata.json"):
                        with open("metadata.json", "r") as f:
                            d = json.load(f)
                    else:
                        d = {} # Start with an empty dict if file doesn't exist

                    # Update keys if input was taken
                    if usrID:
                        d["ACCOUNTID"] = usrID
                    if usrKey:
                        d["APIKEY"] = usrKey
                        
                    # Write back to file
                    with open("metadata.json","w") as f:
                        json.dump(d,f, indent=4)
                    print("Login info saved to metadata.json.")
                except Exception as e:
                    print(f"Warning: Could not save login info to metadata.json. Error: {e}")

        # Adding creating environment to allow for trading
        # Use self.apiKey and self.accountID which are guaranteed to be set if user provided input
        # Added try-except for API initialization, although V20 typically connects later
        try:
            self.api = API(access_token=self.apiKey, environment="practice", headers={"Accept-Datetime-Format": "UNIX"})
            print(f"ForexApi initialized for Account ID: {self.accountID} (Practice Environment).")
        except Exception as e:
            print(f"Error initializing OANDA API: {e}")
            self.api = None # Ensure api is None if initialization fails
    

    def buy_sell(self, pair, units, pip_diff, view=False, terminal_print=True, time_In_Force="FOK",type_="MARKET", price="1"):
        """ 
        :params
            pair - forex pair, ex [EURUSD EUR/USD EUR_USD] are all valid formats
            units - How much to buy. - value makes sell postiion and + makes but position
            view - Doesnt execute the order, just displays the order to fill
        If position is negative, sell pos, else pos buy pos
        """

        p = pair
        if "_" not in p:
            p = pair[:3] + "_" + pair[3:]

        # --- 1. Get Pricing Info with Error Handling ---
        try:
            request = PricingInfo(accountID=self.accountID, params={"instruments": p})
            response = self.api.request(request)
        except V20Error as e:
            print(f"ERROR: OANDA API failed to get pricing for {p}. Ensure pair is valid and credentials are correct. Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching pricing for {p}: {e}")
            return None

        # Extract bid and ask prices from the response
        try:
            prices = response.get("prices", [])
            if not prices:
                print(f"ERROR: No price data found in response for {p}.")
                return None
            
            # Use 'bid' for selling (negative units) and 'ask' for buying (positive units) for better simulation
            # The original code only used 'asks'. Let's keep the original logic but make it robust.
            if units > 0: # Buy
                asset_price = float(prices[0]['asks'][0]['price'])
            else: # Sell
                asset_price = float(prices[0]['bids'][0]['price'])

            pip = 1e-4 if 'EUR' in p or 'GBP' in p or 'AUD' in p else 1e-2 # Standard pip for most pairs
            
            basediff = pip * pip_diff
            
            # Calculate Take Profit (tp) and Stop Loss (sl)
            if units > 0: # Buy position
                tp = asset_price + basediff
                sl = asset_price - basediff
            else: # Sell position
                tp = asset_price - basediff
                sl = asset_price + basediff

        except (IndexError, KeyError, ValueError) as e:
            print(f"ERROR: Failed to parse price data from OANDA response for {p}. Error: {e}")
            return None


        # --- 2. Build Order Info ---
        order_info = {
            "order": {
                "price": price,
                # "takeProfitOnFill": {
                #         "timeInForce": "GTC",
                #         "price": str(int(round(tp,5))) # Rounding for precision
                #     },
                # "stopLossOnFill": {
                #     "timeInForce": "GTC",
                #     "price": str(int(round(sl,5))) # Rounding for precision
                #     },
                
            "timeInForce": time_In_Force, # Use parameter
            "instrument": p,
            "units": str(units),
            "type": type_, # Use parameter
            "positionFill": "DEFAULT"
            }
        }
        

        # if terminal_print:
            # print(json.dumps(order_info, indent=4)) # Prettier print
            # print("ERROR Fulfilling buy/sell")
            # pass
        
        if view:
            return order_info
        else:
            # --- 3. Execute Order with Error Handling ---
            try:
                o = Order.OrderCreate(self.accountID,order_info)
                resp = self.api.request(o)
                print(f"SUCCESS: Order submitted for {units} units of {p}.")
                return resp
            except V20Error as e:
                print(f"ERROR: OANDA API failed to execute order for {p}. Error: {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred while executing order for {p}: {e}")
                return None
            

    def buy_sell_zero_agentic(self, pair, units, pip_diff=10, slope_weight=None, lookback=5, view=False, terminal_print=True):
        """
        Places a 'spread-minimized' order using an agentic approach:
        - Uses recent ticks/candles to find the optimal entry price
        - Computes internal zero-P/L baseline
        - Optionally scales units based on slope_weight
        """

        # --- 1. Format pair ---
        p = pair if "_" in pair else pair[:3] + "_" + pair[3:]

        # --- 2. Fetch recent pricing ---
        try:
            response = self.api.request(PricingInfo(accountID=self.accountID, params={"instruments": p}))
            prices = response.get("prices", [])
            if not prices:
                print(f"ERROR: No prices for {p}")
                return None

            bid = float(prices[0]['bids'][0]['price'])
            ask = float(prices[0]['asks'][0]['price'])
            spread = ask - bid
            mid = (bid + ask)/2
        except Exception as e:
            print(f"ERROR fetching prices for {p}: {e}")
            return None

        # --- 3. Fetch recent candle closes to estimate slope / micro-trend ---
        try:
            candle_resp = self.api.request(instruments.InstrumentsCandles(instrument=p, params={"granularity":"M1","count":lookback}))
            closes = [float(candle['mid']['c']) for candle in candle_resp['candles']]
            slope = closes[-1] - closes[0]  # simple slope over lookback
            if slope_weight is not None:
                units = int(units * slope_weight)
        except Exception as e:
            print(f"WARNING: Could not fetch slope for {p}: {e}")
            slope = 0  # fallback

        # --- 4. Compute agentic optimal entry ---
        # Weighted recent prices, giving more weight to recent ticks
        weights = list(range(1, len(closes)+1))  # older = smaller weight
        weighted_avg = sum(c*w for c,w in zip(closes, weights)) / sum(weights)

        if units > 0:
            optimal_price = min(weighted_avg, ask)  # for buy: can't exceed ask
        else:
            optimal_price = max(weighted_avg, bid)  # for sell: can't go below bid

        # Handle precision
        precision_map = {"JPY":3,"TRY":3,"CHF":5,"USD":5,"EUR":5,"GBP":5,"AUD":5,"NZD":5,"CAD":5,"SGD":4}
        precision = 5
        for k,v in precision_map.items():
            if k in p:
                precision = v
                break
        optimal_price = round(optimal_price, precision)
        pip = 1e-4 if any(x in p for x in ["EUR","GBP","AUD","NZD"]) else 1e-2
        basediff = pip * pip_diff
        tp = optimal_price + basediff if units > 0 else optimal_price - basediff
        sl = optimal_price - basediff if units > 0 else optimal_price + basediff

        # --- 5. Build order ---
        order_info = {
            "order": {
                "instrument": p,
                "units": str(units),
                "price": str(optimal_price),
                "type": "LIMIT",
                "timeInForce": "GTC",
                "positionFill": "DEFAULT",
            }
        }

        if view:
            print(f"[VIEW] {p} optimal price: {optimal_price}, slope={slope}")
            print(json.dumps(order_info, indent=4))
            return order_info

        # --- 6. Execute order ---
        try:
            o = Order.OrderCreate(self.accountID, order_info)
            resp = self.api.request(o)
            fill_price = float(resp["orderFillTransaction"]["price"])
        except Exception as e:
            print(f"ERROR executing order for {p}: {e}")
            return None

        # --- 7. Spread-neutral internal baseline ---
        adjusted_entry = fill_price - (spread/2) if units > 0 else fill_price + (spread/2)

        if terminal_print:
            print(f"AGENTIC EXECUTED: {units} {p} @ {fill_price}")
            print(f"Spread-neutral baseline: {adjusted_entry:.{precision}f}, TP={tp:.{precision}f}, SL={sl:.{precision}f}")

        return {
            "pair": p,
            "units": units,
            "fill_price": fill_price,
            "adjusted_entry": adjusted_entry,
            "spread": spread,
            "tp": tp,
            "sl": sl,
            "slope": slope,
            "response": resp
        }

    def close_all_orders(self, close_positions=False):
        """
        Cancels ALL pending orders (LIMIT, STOP, etc.) for this account.
        Optionally closes all open positions as well.

        :param close_positions: If True, also closes all open positions.
        :return: dict summary of results
        """
        results = {
            "orders_cancelled": [],
            "positions_closed": []
        }

        # --- 1. Fetch All Pending Orders ---
        try:
            list_orders = Order.OrderList(self.accountID)
            response = self.api.request(list_orders)
            orders = response.get("orders", [])
        except Exception as e:
            print(f"ERROR: Could not fetch open orders. Error: {e}")
            return None

        if not orders:
            print("INFO: No pending orders found.")
        else:
            for o in orders:
                order_id = o.get("id")
                instrument = o.get("instrument", "Unknown")
                try:
                    cancel_req = Order.OrderCancel(accountID=self.accountID, orderID=order_id)
                    self.api.request(cancel_req)
                    print(f"SUCCESS: Cancelled order {order_id} for {instrument}.")
                    results["orders_cancelled"].append(order_id)
                except Exception as e:
                    print(f"ERROR: Could not cancel order {order_id} for {instrument}. Error: {e}")

        # --- 2. Optionally Close All Positions ---
        if close_positions:
            try:
                list_positions = Positions.OpenPositions(self.accountID)
                pos_response = self.api.request(list_positions)
                positions = pos_response.get("positions", [])

                for p in positions:
                    instrument = p.get("instrument")
                    print(f"Attempting to close position for {instrument}...")
                    res = self.close(instrument)
                    if res not in (-1, 0, None):
                        results["positions_closed"].append(instrument)

            except Exception as e:
                print(f"ERROR: Failed to fetch or close positions. Error: {e}")

        print(f"SUMMARY: Cancelled {len(results['orders_cancelled'])} orders, Closed {len(results['positions_closed'])} positions.")
        return results


    def get_pair(self, _pair, granularity="M1", onlyPrice=False, as_numpy=True):
        """
        Fetches historical candle data for a pair and returns a NumPy array [Close, High, Low, Open].
        
        Parameters:
            _pair (str): The currency pair (e.g., "EURUSD" or "EUR_USD").
            granularity (str): Candle interval ("M1", "H1", "D1", etc.).
            onlyPrice (bool): If True, returns only the latest close price.
            as_numpy (bool): If True, returns NumPy array with CHLO data.
            
        Returns:
            np.ndarray: [[close, high, low, open], ...] for each candle (oldest → newest)
        """
        p = _pair if "_" in _pair else _pair[:3] + "_" + _pair[3:]

        params = {
            "granularity": granularity,
            "count": 200  # Fetch more candles for better analysis
        }

        try:
            request = instruments.InstrumentsCandles(instrument=p, params=params)
            response = self.api.request(request)
            candles = response.get("candles", [])

            if not candles:
                print(f"WARNING: No candles found for {p}.")
                return None

            # Extract mid prices for each candle
            closes, highs, lows, opens = [], [], [], []
            for c in candles:
                mid = c.get("mid", {})
                closes.append(float(mid.get("c", np.nan)))
                highs.append(float(mid.get("h", np.nan)))
                lows.append(float(mid.get("l", np.nan)))
                opens.append(float(mid.get("o", np.nan)))

            # Return only latest close if requested
            if onlyPrice:
                return closes[-1] if closes else None

            # Create structured NumPy array: columns [Close, High, Low, Open]
            data = np.column_stack([closes, highs, lows, opens])

            if as_numpy:
                return data
            else:
                df = pd.DataFrame(data, columns=["Close", "High", "Low", "Open"])
                return df

        except V20Error as e:
            print(f"ERROR: OANDA API failed to get candle data for {p}. Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching data for {p}: {e}")
            return None




    def close(self, _pair):
        """ Closes ALL open positions for a specific pair"""
        
        pair = (_pair if "_" in _pair else _pair[:3] + "_" + _pair[3:])

        # --- 1. Get Open Positions with Error Handling ---
        try:
            list_orders = Positions.OpenPositions(self.accountID)
            order_dict = self.api.request(list_orders)
        except V20Error as e:
            print(f"ERROR: OANDA API failed to get open positions. Error: {e}")
            return -1
        except Exception as e:
            print(f"An unexpected error occurred while fetching open positions: {e}")
            return -1
            
        plist = order_dict.get('positions', [])
        pair_info = None

        # --- 2. Find Position Info ---
        for i in plist:
            if i['instrument'] == pair:       
                pair_info = i
                break # Found the pair, exit loop
        
        if pair_info is None:
            print(f"INFO: No open position found for {pair}.")
            return -1

        # Determine which side (long/short) to close
        long_units = int(pair_info['long']['units'])
        short_units = int(pair_info['short']['units'])

        toclose = None
        if long_units != 0:
            toclose = {"longUnits" : "ALL"}
        elif short_units != 0:
            toclose = {"shortUnits" : "ALL"}
        else:
            print(f"INFO: Position found for {pair}, but units are 0. Cannot close.")
            return -1
        
        # --- 3. Close Position with Error Handling ---
        try:
            req = Positions.PositionClose(accountID=self.accountID, instrument=pair, data=toclose)
            respo = self.api.request(req)
            print(f"SUCCESS: Closed position for {pair}. Response: {respo.get('orderFillTransaction', {}).get('id', 'N/A')}")
            return respo
        except V20Error as e:
            print(f"ERROR: OANDA API failed to close position for {pair}. Error: {e}")
            return 0
        except Exception as e:
            print(f"An unexpected error occurred while closing position for {pair}: {e}")
            return 0
        
    def view(self,_pair=None,gen_info=False):
        """ Views account info or open positions for a pair """

        # --- 1. Get Positions and Account Details with Error Handling ---
        positions = None
        acc_info = None
        
        try:
            list_orders = Positions.OpenPositions(self.accountID)
            positions = self.api.request(list_orders)
        except V20Error as e:
            print(f"WARNING: OANDA API failed to get open positions. Error: {e}")
        except Exception as e:
            print(f"WARNING: An unexpected error occurred while fetching open positions: {e}")

        try:
            account_info = Account.AccountDetails(self.accountID)
            acc_info = self.api.request(account_info)
        except V20Error as e:
            print(f"WARNING: OANDA API failed to get account details. Error: {e}")
        except Exception as e:
            print(f"WARNING: An unexpected error occurred while fetching account details: {e}")

        # --- 2. Return Requested Info ---
        if acc_info and gen_info:
            return acc_info
        
        if positions is None:
            print("ERROR: Could not fetch open positions.")
            return None # Indicate failure if position fetch failed

        if _pair == None:
            return positions

        else:
            pair = (_pair if "_" in _pair else _pair[:3] + "_" + _pair[3:]) 

            for i in positions.get('positions', []):
                if i['instrument'] == pair:
                    return i
            # Return None if specific pair position is not found
            return None
    
    
    def log_info(self,log_off=False):
        """ Logs account balance and P&L to pricelog.csv """
        if log_off:
            return

        # --- 1. Ensure file exists ---
        try:
            if not os.path.exists("pricelog.csv"):
                with open("pricelog.csv", "w") as f:
                    f.write("Time,Bal,Pl\n")
        except Exception as e:
            print(f"ERROR: Could not write to pricelog.csv. Logging skipped. Error: {e}")
            return
            
        # --- 2. Get Account Details with Error Handling ---
        response = None
        try:
            bal_req = Account.AccountDetails(self.accountID)
            response = self.api.request(bal_req)
        except V20Error as e:
            print(f"WARNING: OANDA API failed to get account details for logging. Error: {e}")
            return
        except Exception as e:
            print(f"WARNING: An unexpected error occurred while fetching account details for logging: {e}")
            return

        # --- 3. Extract and Log Data ---
        try:
            a = response['account']

            bal =  float(a.get('balance', 0.0))
            pl = float(a.get('pl', 0.0))
            t = datetime.datetime.now().timestamp()

            with open("pricelog.csv", "a") as f:
                f.write(f"{t},{bal},{pl}\n")
            
        except (KeyError, ValueError) as e:
            print(f"ERROR: Failed to parse account details for logging. Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during logging: {e}")
            
        return
        
    def bal(self, terminal_print=True):
        """
        Retrieves and displays the current account balance, NAV, and P/L.
        """
        try:
            r = Account.AccountSummary(self.accountID)
            response = self.api.request(r)
            summary = response.get('account', {})

            balance = float(summary.get('balance', 0))
            nav = float(summary.get('NAV', 0))
            unrealizedPL = float(summary.get('unrealizedPL', 0))
            realizedPL = float(summary.get('pl', 0))
            marginUsed = float(summary.get('marginUsed', 0))
            marginAvail = float(summary.get('marginAvailable', 0))

            if terminal_print:
                print("\n--- ACCOUNT BALANCE SUMMARY ---")
                print(f"Account ID:     {self.accountID}")
                print(f"Balance:        {balance:,.2f}")
                print(f"NAV:            {nav:,.2f}")
                print(f"Unrealized P/L: {unrealizedPL:,.2f}")
                print(f"Realized P/L:   {realizedPL:,.2f}")
                print(f"Margin Used:    {marginUsed:,.2f}")
                print(f"Margin Avail:   {marginAvail:,.2f}")
                print("--------------------------------")

            return {
                "balance": balance,
                "NAV": nav,
                "unrealizedPL": unrealizedPL,
                "realizedPL": realizedPL,
                "marginUsed": marginUsed,
                "marginAvailable": marginAvail,
            }

        except Exception as e:
            print(f"ERROR: Failed to fetch account balance: {e}")
            return None
