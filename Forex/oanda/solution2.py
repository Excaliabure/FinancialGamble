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

min = fx.min("EUR_USD").to_numpy()[0][:,2]
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
        dm = fx.algo.deriv12(min)
        dh = fx.algo.deriv12(hr)
        dd = fx.algo.deriv12(day)
    

        if dm ==dh  and prevDecision != dm:
            prevDecision = dm
            env.close("EUR_USD")
            env.buy_sell("EUR_USD",1000 * prevDecision, 40)
            data_arr_collection('data.json', 'bal', float(env.view(gen_info=True)['account']['balance']))

        time.sleep(30)
