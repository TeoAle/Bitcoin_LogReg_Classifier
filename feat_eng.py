import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime


if __name__ == "__main__":
    
    # --------------------- Importing the Dataset downloaded from Binance of the price of bitcoin --------------------- 
    
    data_ohlc = pd.read_csv('BTCUSDT-1m-2022-03.csv', 
                            names = ['open time', 'open', 'high', 'low', 'close', 'volume', 'close time', 'quote asset volume', 'number of       trades', 'taker buy base asset volume', 'taker buy quote asset volume', 'ignore'])


    data_ohlc['timestamp'] = data_ohlc['open time'].map(lambda t: datetime.fromtimestamp(t/1000)) # Open time in Binance is in ms so i divide by 1000
    data_ohlc = data_ohlc.set_index('timestamp')                                                  
    print(data_ohlc.head())

    # --------------------- Checking if the data are fine by plotting them ---------------------

    data_ohlc['ratio'] = np.concatenate([(data_ohlc['close'].iloc[1:].values / data_ohlc['close'].iloc[:-1]), [1]])

    data_ohlc.ratio.hist(bins=100).plot() # The logaritmic return of the price is symmetrical with respect to zero so the data are balanced
    data_ohlc.close.plot()                # Plotting the closes we see that the data is not just rising or just falling in the period we chose

    # --------------------- Creating the target column (label) ---------------------


    treshold = 0.0004               # I prefer to use a fixed treshold instead of a treshold based on volatility
    data_ohlc['label'] = 0
    data_ohlc.loc[data_ohlc['ratio'] > 1 + treshold, 'label'] = 1
    data_ohlc.loc[data_ohlc['ratio'] < 1 - treshold, 'label'] = -1

    data_ohlc.label.value_counts() # The classes aren't quite balanced so i will manage oversampling with SMOTE

    # --------------------- Creating the features (trading indicators) on which i will train the ML model ---------------------
    
    data_ohlc['volatility'] = data_ohlc['close'].ewm(span=100).std()

    data_ohlc['mid'] = (data_ohlc['high'] + data_ohlc['low'])/2

    data_ohlc['excursion'] = (data_ohlc['high'] - data_ohlc['low'])/data_ohlc['mid']

    # Creating the Balance of Power (BOP) oscillator
    data_ohlc['BOP'] = (data_ohlc['close'] - data_ohlc['open'])/(data_ohlc['high'] - data_ohlc['low'])

    data_ohlc['momentum'] = (data_ohlc['close'] - data_ohlc['mid'])/(data_ohlc['high'] - data_ohlc['low'])

    # Creating the Money Flow Index
    data_ohlc['typical_price'] = (data_ohlc['high'] + data_ohlc['low'] + data_ohlc['close'])/3
    data_ohlc['money_flow'] = data_ohlc['typical_price']*data_ohlc['volume']

    # Creating the Moving Average Convergence Divergence indicator
    data_ohlc['MACD'] = data_ohlc.close.rolling(5).mean()/data_ohlc.close.rolling(20).mean() - 1

    # Creating a Stochastic Oscillator
    k_period = 14  # Define periods
    d_period = 3

    data_ohlc['n_high'] = data_ohlc['high'].rolling(k_period).max() # Adds a "n_high" column with max value of previous 14 periods

    data_ohlc['n_low'] = data_ohlc['low'].rolling(k_period).min() # Adds an "n_low" column with min value of previous 14 periods

    data_ohlc['perc_K'] = (data_ohlc['close'] - data_ohlc['n_low']) * 100 / (data_ohlc['n_high'] - data_ohlc['n_low']) # Uses the min/max values to calculate the %k (as a percentage)
    
    data_ohlc['perc_D'] = data_ohlc['perc_K'].rolling(d_period).mean() # Uses the %k to calculates a SMA (Simple Moving Average) over the past 3 values of %k
    
    # --------------------- Cleaning the dataset dropping rows with NaN (I have enough rows to do this without reducing too much my dataset)  ---------------------
    
    data_ohlc.isna().sum() # taking a look at the columns with NaN (as expected just the columns created before and not imported in the original dataset)
    
    data_ohlc.dropna(axis=0, inplace=True)
    
    # --------------------- Creating a new cleaned dataset with just the columns i'm interested in ---------------------
    data_ohlc[['excursion','BOP','momentum','typical_price','money_flow','MACD','perc_K','perc_D','volume','number of trades','taker buy base asset volume', 'volatility','label']].to_csv('BTCUSDT_CLEANED.csv')
    
    
