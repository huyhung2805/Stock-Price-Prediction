import pandas as pd
import numpy as np
import ta
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from ta.momentum import WilliamsRIndicator
from ta.trend import ADXIndicator, PSARIndicator

def add_technical_indicators(data):
    data = data.copy()
    
    # Ensure datetime index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    elif not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    if len(data) < 50:
        return data

    # Price-based features
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log1p(data['Returns'])
    data['True_Range'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    
    # Moving Averages
    for window in [5, 10, 20, 50, 100, 200]:
        data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
        data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
        # Distance from MA
        data[f'Distance_from_SMA_{window}'] = (data['Close'] - data[f'SMA_{window}']) / data[f'SMA_{window}']
    
    # Volatility Indicators
    for window in [7, 14, 21]:
        data[f'Volatility_{window}d'] = data['Returns'].rolling(window=window).std()
        
    # Volume Indicators
    data['Volume_SMA_5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
    data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    data['MFI'] = MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
    
    # Momentum Indicators
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['Williams_R'] = WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    
    # Trend Indicators
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    
    adx = ADXIndicator(data['High'], data['Low'], data['Close'])
    data['ADX'] = adx.adx()
    data['ADX_Pos'] = adx.adx_pos()
    data['ADX_Neg'] = adx.adx_neg()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()
    data['BB_Middle'] = bb.bollinger_mavg()
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Support and Resistance
    data['Support_Level'] = data['Low'].rolling(window=20).min()
    data['Resistance_Level'] = data['High'].rolling(window=20).max()
    
    # Advanced Features
    data['Price_Range'] = data['High'] - data['Low']
    data['Price_Range_Ratio'] = data['Price_Range'] / data['Close']
    data['Gap'] = data['Open'] - data['Close'].shift(1)
    
    # Seasonal Features
    data['Day_sin'] = np.sin(data.index.dayofweek * (2 * np.pi / 7))
    data['Day_cos'] = np.cos(data.index.dayofweek * (2 * np.pi / 7))
    data['Month_sin'] = np.sin((data.index.month - 1) * (2 * np.pi / 12))
    data['Month_cos'] = np.cos((data.index.month - 1) * (2 * np.pi / 12))
    data['Quarter'] = data.index.quarter
    
    # Fill missing values
    data = data.bfill().ffill()
    
    return data