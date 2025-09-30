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
from scipy.ndimage import gaussian_filter1d


from oandapyV20 import API
apiKey = None
accountID = None
with open("dev_settings.json", "r") as file:
    d = json.load(file)
    apiKey = d["acckey"]
    accountID = d["accid"]



##################################################################

#######################################################

pcp = np.load("pcp.npy")# 155 x 4050

x = np.zeros((6,1)); x[0,:] = -1; x[-1,:] = 1
monies = np.zeros((pcp.shape[0],1)) + 100

# print(pcp.shape, x.shape)
viewarr = []


def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

WINDOWSIZE = 3

n = np.zeros((pcp.shape[0],pcp.shape[1]))

def ema(arr, span):
    """
    Compute the Exponential Moving Average (EMA) of a 1D array.

    Parameters
    ----------
    arr : array_like
        Input data (1D).
    span : int
        Span for the EMA (controls smoothing, larger = smoother).

    Returns
    -------
    ema_values : np.ndarray
        The EMA of the input.
    """
    arr = np.asarray(arr, dtype=float)
    alpha = 2 / (span + 1)
    
    ema_values = np.zeros_like(arr)
    ema_values[0] = arr[0]
    for i in range(1, len(arr)):
        ema_values[i] = alpha * arr[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values

for i in range(pcp.shape[0]):

    n[i] = ema(pcp[i],3)


pcp = pcp[:,pcp.shape[1]-150:]
# pcp = pcp[139]

print(pcp)
plt.plot(pcp)
plt.show()


# Z = pcp[122]

# plt.plot(Z)
# plt.show()

# x = np.arange(Z.shape[1])   # columns
# y = np.arange(Z.shape[0])   # rows
# X, Y = np.meshgrid(x, y)

# # 3D surface plot
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# ax.set_xlabel("X-axis (columns)")
# ax.set_ylabel("Y-axis (rows)")
# ax.set_zlabel("Z value")
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.show()


# Clean up pairs



for i in range(6, pcp.shape[1]-12,6):

    a =  pcp[:,i:i+6]
    ans = a @ x / 6 * 100
    
    decision = monies * ans

    

    incvec = (pcp[:,i+7] - pcp[:,i+6]) /2 * 100 
    print(sum(monies))

    monies = monies - decision




    profit = decision + (decision * np.expand_dims(incvec, axis=1))
    



    viewarr.append(monies.sum())

# plt.plot(pcp[2])
# plt.show()
plt.plot(monies)
plt.show()







