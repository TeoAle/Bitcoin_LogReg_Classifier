import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression # model we are going to use
from datetime import datetime
from Class_ML.ML_Pipeline import MLPipeline # this is the class for the ML Pipeline I created in the folder Class_ML 

warnings.filterwarnings("ignore")

########################################## MODEL EVALUATION ##########################################


if __name__ == "__main__":
  
      # --------------------- Importing the cleaned dataset we created in 'feature_engineering.py' ---------------------

      data_clean = pd.read_csv('data/BTCUSDT_CLEANED.csv')
      data_clean = data_clean.set_index('timestamp')

      # --------------------- Evaluating the model through our Machine Learning Pipeline Class ---------------------

      # Features we are going to use for prediction

      features = ['excursion',
                  'BOP',
                  'momentum',
                  'typical_price',
                  'money_flow',
                  'MACD',
                  'perc_K',
                  'perc_D',
                  'volume',
                  'number of trades',
                  'taker buy base asset volume', 
                  'volatility']
      
      target_array = data_clean[['label']].values

      model = LogisticRegression(random_state=42)
      parameters = {                                       # We give a range of reasonable parameters to be tested through the Grid Search
                    'C': [0.001,0.01,0.1,1,10,100],        # (in the parameter_tuning function) of the MLPipeline class
                    'class_weight' : [None, 'Balanced'],
                    'max_iter' : [100,1000,10000]
                    }


      pipeline = MLPipeline(model=model)
      pipeline.load_data(dataset=data_clean[features],target=target_array)
      pipeline.split_data()
      pipeline.preprocessing()
      pipeline.parameter_tuning(parameters)
      pipeline.model_evaluation()
      pipeline.k_fold_cross_validation()
      pipeline.report()

  
  
