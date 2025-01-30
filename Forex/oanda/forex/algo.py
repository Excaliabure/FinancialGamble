
from .utilities._utils import deriv
import numpy as np
import matplotlib.pyplot as plt


None
"""
Used to store written algorithms 

"""
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



"""
Custom Algorithms

"""


def deriv12(arr, raw_data=False,plot=True, tol=1e-6):

    """
    Given a 1d array, output y'' - y- = 0 and buy if y' > 0, sell y' < 0

    Note this has na ssociated ai.derv12() in the works

    :params
        arr - 1d iterable element. Only accepts real numbers as integers
        * raw_data - return the buy,sell,tbuy,tsell if set to True
        * tol - tolerance for abs(y''-y') < tol
    :out

        tbuy tsell buy sell - 1d arrays with tbuy & tsell as time and buy/sell as the val
    
    """
    
    smoothArr = arr.copy()
    
    smoothArr[0] = arr[0]
    for j in range(6):
        for i in range(2,len(arr)):
            mid = (arr[i] - arr[i-1])/2
            smoothArr[i-1] += mid
            smoothArr[i] -= mid
    
    y = smoothArr    
    dy = deriv(y)
    ddy = deriv(dy)

    buy = []
    sell = []
    tbuy = []
    tsell = []
    seg = ddy - dy
    collision,tcollision = [], []

    for i in range(1,len(dy)):
        if seg[i-1] > 0 and seg[i] < 0 or seg[i-1] < 0 and seg[i] > 0:
            collision.append(0)
            tcollision.append(i)

    for i in tcollision:
        if dy[i] < 0:
            sell.append(y[i])
            tsell.append(i)
        else:
            buy.append(y[i])

            tbuy.append(i)

        
        
    if raw_data:
        return buy, sell, tbuy, tsell
    
    if len(tbuy) == 0 or len(tsell) == 0:
        return 1
    elif tbuy[-1] > tsell[-1]:
        return 1
    else:
        return -1