import forex as fx
import json
import time
import numpy as np
import pandas_ta as ta

import pandas as pd
from datetime import datetime, timedelta
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Candle fetch
def fetch_candles_for_supertrend(OANDA_API_KEY, OANDA_INSTRUMENT):
    client = oandapyV20.API(access_token=OANDA_API_KEY)

    # 4 hours ago to now
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    params = {
        "from": start_time.isoformat() + "Z",
        "to": end_time.isoformat() + "Z",
        "granularity": "S5",  # 1-minute intervals
        "price": "M",         # Midpoint pricing
    }

    r = instruments.InstrumentsCandles(instrument=OANDA_INSTRUMENT, params=params)
    client.request(r)
    candles = r.response['candles']

    data = []
    for c in candles:
        if c['complete']:
            time = c['time']
            o = float(c['mid']['o'])
            h = float(c['mid']['h'])
            l = float(c['mid']['l'])
            close = float(c['mid']['c'])
            data.append([time, o, h, l, close])

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.to_csv("supertrend_data_4h.csv")

    print("Saved 4 hours of 1-minute candles to 'supertrend_data_4h.csv'.")
    return df




if __name__ == '__main__':
    
    from oandapyV20 import API
    apiKey = None
    accountID = None
    with open("dev_settings.json", "r") as file:
        d = json.load(file)
        apiKey = d["acckey"]
        accountID = d["accid"]
    


    env = fx.ForexApi(apikey=apiKey, accountid=accountID)
    data = fetch_candles_for_supertrend(apiKey, "EUR_USD")
    su = ta.supertrend(data['high'], data['low'], data['close'], length=10, multiplier=3)
    
    arr = su["SUPERTd_10_3"].to_numpy() 
    
    pos = arr[-1]
    newpos = arr[-1]
    
    env.close("EUR_USD")
    env.buy_sell("EUR_USD", -10000 * pos,  99)
            
    while True:

        data = fetch_candles_for_supertrend(apiKey, "EUR_USD")
        su = ta.supertrend(data['high'], data['low'], data['close'], length=10, multiplier=3)
        arr = su["SUPERTd_10_3"].to_numpy() * -1
        print(data)

        newpos = arr[-1]

        if pos != newpos:

            pos = newpos

            env.close("EUR_USD")
            time.sleep(1)
            print(arr)
            env.buy_sell("EUR_USD",1000 * pos,  99)


        time.sleep(10)
        
