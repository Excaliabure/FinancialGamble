import forex as fx
import json
import time


from oandapyV20.endpoints.pricing import PricingInfo
import oandapyV20.endpoints.positions as Positions
import oandapyV20.endpoints.accounts as Account

import oandapyV20.endpoints.orders as Order
from oandapyV20 import API
import datetime

# Step 1

minute = fx.min("EUR_USD").to_numpy()[0][:,2]
hr = fx.hr("EUR_USD").to_numpy()[0][:,2]
day = fx.day("EUR_USD").to_numpy()[0][:,2]


# print(f"Min {u(m)}\nHour {u(h)}\nDay {u(d)}")

def data_arr_collection(filename,name,val):

    # if '.json' not in file:
    #     file = file + '.json'

    with open(filename, 'r') as file:
        data = json.load(file)
        # print("Data read from file:", data)

    if name not in data.keys():
        data[name] = []
    data[name].append(val)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4) 

if __name__ == '__main__':
    
    from oandapyV20 import API
    apiKey = None
    accountID = None
    with open("dev_settings.json", "r") as file:
        d = json.load(file)
        apiKey = d["acckey"]
        accountID = d["accid"]
    
    
    from oandapyV20.endpoints.pricing import PricingInfo
    env = fx.ForexApi(apiKey,accountID)
    prevDecision = 0
    while True:
        
        minute = fx.min("EUR_USD").to_numpy()[0]
        hr = fx.hr("EUR_USD").to_numpy()[0]
        day = fx.day("EUR_USD").to_numpy()[0]
        time.sleep(1)

        if minute != None and hr != None and day != None:
            minute = minute[:,2]
            hr = hr[:,2]
            day = day[:,2]

            dm = fx.algo.deriv12(minute)
            dh = fx.algo.deriv12(hr)
            dd = fx.algo.deriv12(day)
        


            if dm == dh  and prevDecision != dh:
                print(f"{'Bought' if dh < 0 else 'Sold'}")
                env.close("EUR_USD")
                time.sleep(2)
                if prevDecision != 0:
                    env.buy_sell("EUR_USD",-1000 * dh, 300)
                
                prevDecision = dh
                data_arr_collection('data.json', 'bal', float(env.view(gen_info=True)['account']['balance']))

            while (env.view("EUR_USD") == None and prevDecision != 0):
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

        
