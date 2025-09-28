import forex as fx
import json
import time
import numpy as np
import pandas_ta as ta

import pandas as pd
from datetime import datetime, timedelta
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import os
import matplotlib.pyplot as plt



from oandapyV20 import API
apiKey = None
accountID = None
with open("dev_settings.json", "r") as file:
    d = json.load(file)
    apiKey = d["acckey"]
    accountID = d["accid"]


class ev:

    def __init__(self):

        self.pcp = np.load("pcp.npy")# 155 x 4050
        self.idx = 0
        self.forward = 6

    def step(self,inc=6):
        
        temp = self.forward
        self.forward += inc
        if self.idx+inc >= len(self.pcp):
            return None
        return self.pcp[:,self.idx:self.idx+self.forward]
        
    


env = ev()
money = 10000



def percent_incdec(arr):
    if arr[0] == 0:
        return 0
        
    r  = (arr[-1] - arr[0]) / arr[0] * 10000
    
    return r



running = True
inc = env.step()
slopes = np.zeros((inc.shape[0], 4050 // 6))

time_step = 0

while running:
    inc = env.step()
    
    for i in range(inc.shape[0]):
        s = percent_incdec(inc[i])
        for j in range(len(inc[i])):
            if s != 0:
                break
            else:
                s = percent_incdec(inc[j])
        # print(s)
        slopes[time_step][i] = s
        time_step += 1

    print(slopes.max())
    print(slopes.shape)
    if type(inc) != np.array:
        running=False

