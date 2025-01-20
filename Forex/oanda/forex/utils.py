import numpy as np
import json
import datetime

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

def integral(iterable, dx=1):
    """ 
    This is more like the anti derivative than the integral
    Note: Use sum(integral(arr)) to get teh valued integral
    """

    integral = []
    
    for i in range(1, len(iterable)):
        # Calculate the area of each trapezoid
        integral.append((iterable[i - 1] + iterable[i]) * dx / 2)

    return integral


def to_date(stamp):
    return datetime.datetime.fromtimestamp(stamp)



def read_settings():
    
    with open("metadata.json", "r") as f:
        return json.load(f)
    return None


def _json_save(file,key,data):

    with open(file, 'r') as file:
        data = json.load(file)

    if key not in data.keys():
        data[key] = []
    data[key].append(data)
    with open(file, 'w') as file:
        json.dump(data, file, indent=4) 


class utils:
    
    @staticmethod
    def calculate_ema(prices, period):
        
        ema = []
        k = 2 / (period + 1)
        ema.append(prices[0])  # Start with the first closing price as the first EMA

        for i in range(1, len(prices)):
            ema.append(prices[i] * k + ema[i-1] * (1 - k))

        return np.array(ema)
    
    @staticmethod
    def integral(iterable, dx=1):
        """ 
        This is more like the anti derivative than the integral
        Note: Use sum(integral(arr)) to get teh valued integral
        """

        integ = []
    
        for i in range(1, len(iterable)):
            # Calculate the area of each trapezoid
            integ.append((iterable[i - 1] + iterable[i]) * dx / 2)

        return integ

        
    
