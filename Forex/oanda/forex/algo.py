
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
    
def ema(series, period):
    """Compute Exponential Moving Average (EMA) using NumPy."""
    alpha = 2 / (period + 1)
    ema_values = np.zeros_like(series)
    ema_values[0] = series[0]  # Initialize first EMA value
    for i in range(1, len(series)):
        ema_values[i] = alpha * series[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values


def macd(ohlc, fast_period=12, slow_period=26, signal_period=9):
    """
    Compute MACD from OHLC candlestick data.

    Parameters:
    - ohlc (numpy.ndarray): 2D array with shape (N, 4) containing Open, High, Low, Close prices.
    - fast_period (int): EMA fast period (default 12).
    - slow_period (int): EMA slow period (default 26).
    - signal_period (int): EMA signal line period (default 9).

    Returns:
    - macd (numpy.ndarray): MACD line values.
    - signal (numpy.ndarray): Signal line values.
    - histogram (numpy.ndarray): MACD Histogram values.
    """
    close_prices = ohlc[:, 3]  # Extract Close prices
    N = len(close_prices)

    if N < slow_period:  
        raise ValueError("Not enough data points for slow EMA calculation")

    # Compute MACD Line: (Fast EMA - Slow EMA)
    ema_fast = ema(close_prices, fast_period)
    ema_slow = ema(close_prices, slow_period)
    macd = ema_fast - ema_slow

    # Compute Signal Line (9-period EMA of MACD)
    signal = ema(macd, signal_period)

    # Compute Histogram (MACD - Signal)
    histogram = macd - signal

    return macd, signal, histogram

def smi(chlo, period=14, smooth1=3, smooth2=3):
    """
    Compute Stochastic Momentum Index (SMI) using CHLO (Close, High, Low, Open) data.

    Parameters:
    - chlo (numpy.ndarray): 2D array with shape (N, 4) containing Close, High, Low, Open prices.
    - period (int): Lookback period for high-low range (default 14).
    - smooth1 (int): Smoothing period for D1 (default 3).
    - smooth2 (int): Smoothing period for D2 (default 3).

    Returns:
    - smi (numpy.ndarray): Stochastic Momentum Index values.
    - smi_signal (numpy.ndarray): Signal line values.
    """
    close_prices = chlo[:, 0]
    high_prices = chlo[:, 1]
    low_prices = chlo[:, 2]
    N = len(close_prices)

    if N < period:
        raise ValueError("Not enough data points for SMI calculation")

    # Compute HL Midpoint
    hl_mid = (np.convolve(high_prices, np.ones(period)/period, mode='valid') + 
              np.convolve(low_prices, np.ones(period)/period, mode='valid')) / 2

    # Compute Distance
    distance = close_prices[period - 1:] - hl_mid

    # Compute EMA of Distance (D1 and D2)
    d1 = ema(distance, smooth1)
    d2 = ema(np.abs(distance), smooth2)

    # Compute SMI
    smi = np.where(d2 != 0, 100 * (d1 / d2), 0)  # Avoid division by zero

    # Compute Signal Line (EMA of SMI)
    smi_signal = ema(smi, smooth1)

    return smi / 100, smi_signal / 100


def rsi(chlo, period=14):
    """Compute Relative Strength Index (RSI) using CHLO data."""
    close_prices = chlo[:, 0]
    delta = np.diff(close_prices)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return np.concatenate(([50] * (period - 1), rsi)) / 100  # Padding for the first period-1 values

def bollinger_bands(chlo, period=20, num_std=2):
    """Compute Bollinger Bands using CHLO data."""
    close_prices = chlo[:, 0]
    
    # Simple moving average (SMA)
    sma = np.convolve(close_prices, np.ones(period)/period, mode='valid')
    
    # Calculate standard deviation
    rolling_std = np.array([np.std(close_prices[i-period+1:i+1]) for i in range(period-1, len(close_prices))])
    
    # Upper and Lower Bands
    upper_band = sma + (num_std * rolling_std)
    lower_band = sma - (num_std * rolling_std)
    
    return upper_band, sma[period-1:], lower_band


def compute_supertrend(chlo, period=14, multiplier=3):
    """Compute SuperTrend Indicator using CHLO data."""
    high_prices = chlo[:, 1]
    low_prices = chlo[:, 2]
    close_prices = chlo[:, 0]
    
    # ATR Calculation (Average True Range)
    atr = np.array([np.max([high_prices[i] - low_prices[i], 
                            abs(high_prices[i] - close_prices[i-1]), 
                            abs(low_prices[i] - close_prices[i-1])]) for i in range(1, len(chlo))])
    
    atr = np.concatenate(([0] * (period-1), np.convolve(atr, np.ones(period)/period, mode='valid')))
    
    # Calculate Basic Bands
    hl_avg = (high_prices + low_prices) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize the Supertrend array
    supertrend = np.zeros_like(close_prices)
    
    # Set the first Supertrend value based on the price direction
    supertrend[period] = upper_band[period] if close_prices[period] > upper_band[period] else lower_band[period]

    # Apply trend condition and update Supertrend value
    for i in range(period + 1, len(close_prices)):
        if close_prices[i] > supertrend[i-1]:
            supertrend[i] = upper_band[i]  # Trend is Up
        else:
            supertrend[i] = lower_band[i]  # Trend is Down

    return supertrend



# def vwap(chlo, volume, period=14):
#     """Compute Volume Weighted Average Price (VWAP) using CHLO data."""
#     close_prices = chlo[:, 0]
#     high_prices = chlo[:, 1]
#     low_prices = chlo[:, 2]
    
#     # Typical Price = (High + Low + Close) / 3
#     typical_price = (high_prices + low_prices + close_prices) / 3
    
#     # Cumulative VWAP calculation
#     cumulative_tp = np.cumsum(typical_price * volume)
#     cumulative_vol = np.cumsum(volume)
    
#     vwap = cumulative_tp / cumulative_vol
    
#     return vwap