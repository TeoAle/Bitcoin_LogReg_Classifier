import numpy as np
import pandas as pd
from datetime import datetime
from fastparquet import write
import matplotlib.pyplot as plt
import seaborn as sns
from Class_ML.ML_Pipeline import MLPipeline # this is the class for the ML Pipeline I created in the folder Class_ML 
from sklearn.linear_model import LogisticRegression # model we are going to use
import warnings
warnings.filterwarnings("ignore")

############################ FEATURE ENGINEERING & MODEL EVALUATION ############################

# --------------------- Defining the '_get_volatility' function, we will use it to create a feature  --------------------- 

def _get_volatility(prices, span=100, delta=pd.Timedelta(hours=1)): 
    """
    Compute price return of the form p[t]/p[t-1] - 1
    
    Input: prices :: pd series of prices
           span0  :: the width or lag of the ewm() filter
           delta  :: time interval of volatility to be computed
    Output: pd series of volatility for each given time interval
    """
    
    # find p[t-1] indices given delta
    df0 = prices.index.searchsorted(prices.index - delta)
    df0 = df0[df0 > 0]  
    
    # align p[t-1] timestamps to p[t] timestamps 
    df0 = pd.Series(prices.index[df0 - 1],
                   index=prices.index[prices.shape[0] - df0.shape[0] : ])
    
    # get values for each timestamps then compute returns
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values - 1
    
    # estimate rolling standard deviation
    df0 = df0.ewm(span=span).std()
    df0 = df0[df0 != 0]
    
    return df0
  
  
# --------------------- Importing the Dataset downloaded from Binance of the price of bitcoin --------------------- 

data_ohlc = pd.read_csv('BTCUSDT-1m-2022-03.csv', 
                        names = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])


data_ohlc['timestamp'] = data_ohlc['Open time'].map(lambda t: datetime.fromtimestamp(t/1000)) # Open time in Binance is in ms so i divide by 1000
data_ohlc = data_ohlc.set_index('timestamp')                                                  
print(data_ohlc.head())

# --------------------- Checking if the data are fine by plotting them ---------------------

data_ohlc['Ratio'] = np.concatenate([(data_ohlc['Close'].iloc[1:].values / data_ohlc['Close'].iloc[:-1]), [1]])

data_ohlc.Ratio.hist(bins=100).plot() # The logaritmic return of the price is symmetrical with respect to zero so the data are balanced
data_ohlc.Close.plot()                # Plotting the closes we see that the data is not just rising or just falling in the period we chose

# --------------------- Creating the target column (label) ---------------------

data_ohlc['Volatility'] = _get_volatility(data_ohlc['Close'],delta=pd.Timedelta(hours=2))

data_ohlc['label'] = 0
data_ohlc.loc[data_ohlc['Ratio'] > 1 + 0.1*data_ohlc['volatility'], 'label'] = 1
data_ohlc.loc[data_ohlc['Ratio'] < 1 - 0.1*data_ohlc['volatility'], 'label'] = -1

data_ohlc.label.value_counts() # The classes aren't quite balanced so i will manage oversampling with SMOTE

# --------------------- Creating the features (trading indicators) on which i will train the ML model ---------------------


data_ohlc['Mid'] = (data_ohlc['High'] + data_ohlc['Low'])/2

data_ohlc['Excursion'] = (data_ohlc['High'] - data_ohlc['Low'])/data_ohlc['Mid']

# Creating the Balance of Power (BOP) oscillator
data_ohlc['BOP'] = (data_ohlc['Close'] - data_ohlc['Open'])/(data_ohlc['High'] - data_ohlc['Low'])

data_ohlc['Momentum'] = (data_ohlc['Close'] - data_ohlc['Mid'])/(data_ohlc['High'] - data_ohlc['Low'])

# Creating the Money Flow Index
data_ohlc['Typical_price'] = (data_ohlc['High'] + data_ohlc['Low'] + data_ohlc['Close'])/3
data_ohlc['Money_flow'] = data_ohlc['Typical_price']*data_ohlc['Volume']

# Creating the Moving Average Convergence Divergence indicator
data_ohlc['MACD'] = data_ohlc.Close.rolling(5).mean()/data_ohlc.Close.rolling(20).mean() - 1

# Creating a Stochastic Oscillator
k_period = 14  # Define periods
d_period = 3

data_ohlc['n_high'] = data_ohlc['High'].rolling(k_period).max() # Adds a "n_high" column with max value of previous 14 periods

data_ohlc['n_low'] = data_ohlc['Low'].rolling(k_period).min() # Adds an "n_low" column with min value of previous 14 periods

data_ohlc['%K'] = (data_ohlc['Close'] - data_ohlc['n_low']) * 100 / (data_ohlc['n_high'] - data_ohlc['n_low']) # Uses the min/max values to calculate the %k (as a percentage)

data_ohlc['%D'] = data_ohlc['%K'].rolling(d_period).mean() # Uses the %k to calculates a SMA (Simple Moving Average) over the past 3 values of %k

# --------------------- Cleaning the dataset dropping rows with NaN (I have enough rows to do this without reducing too much my dataset)  ---------------------

data_ohlc.isna().sum() # taking a look at the columns with NaN (as expected just the columns created before and not imported in the original dataset)

data_ohlc = data_ohlc.dropna(axis=0)

# --------------------- Evaluating the model through our Machine Learning Pipeline Class ---------------------

# The features we are going to use for the prediction are:
# - Excursion
# - BOP
# - Momentum
# - Typical_price
# - Money_flow
# - MACD
# - %K
# - %D
# - Volume
# - Number of trades
# - Taker buy base asset volume
# - Volatility

# We already manually normalized some of them, the StandarScaler will do the same with the others

features = ['Excursion','BOP','Momentum','Typical_price','Money_flow','MACD','%K','%D','Volume','Number of trades','Taker buy base asset volume', 'Volatility']

model = LogisticRegression(random_state=42)
parameters = {                                     # We give a range of reasonable parameters to be tested through the Grid Search
              'C': [0.001,0.01,0.1,1,10,100],      # (in the parameter_tuning function) of the MLPipeline class
              'class_weight' : [None, 'Balanced'],
              'max_iter' : [100,1000,10000]
             }

pipeline = MLPipeline(model=model, data=data_ohlc, features=features)
pipeline.load_data()
pipeline.split_data()
pipeline.preprocessing()
pipeline.parameter_tuning(parameters)
pipeline.model_evaluation()
pipeline.k_fold_cross_validation()
pipeline.markdown_report()


