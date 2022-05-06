import numpy as np
import pandas as pd
from datetime import datetime
from fastparquet import write
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler #for Scaling the features
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from IPython.display import display, Markdown 
import pickle
from sklearn.metrics import plot_confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


# --------------------  Creating the Machine Learning pipeline useful for every dataset and classifier (the target must be named 'label') --------------------

class MLPipeline():
    
    def __init__(self,model,data:pd.DataFrame,features:list,test_size=0.3,k_folds=10):
        self.model = model
        self.model_name = type(model).__name__
        self.data = data
        self.features = features
        self.test_size = test_size
        self.k_folds = k_folds
        self.scaler = None

    def load_data(self): 
        self.X,self.y = self.data[self.features],self.data[['label']].values


    def split_data(self):
        # Splitting train and test data 
        self.X_train_0, self.X_test, self.y_train_0, self.y_test = \
            train_test_split(self.X,self.y, test_size=self.test_size, stratify=self.y, random_state=1)
        # balancing classes with SMOTE
        sampler = SMOTE(random_state=0)
        self.X_train, self.y_train = sampler.fit_resample(self.X_train_0, self.y_train_0)
        
        
    def preprocessing(self):
        # Scaling features
        self.scaler = StandardScaler()
        self.X_train =self.scaler.fit_transform(self.X_train)              
        self.X_test =self.scaler.transform(self.X_test) 
        
        
    def parameter_tuning(self, parameters):
        # GridSearchCV to find optimal parameters
        grid = GridSearchCV(self.model, parameters, cv=self.k_folds)
        grid.fit(self.X_train, self.y_train)

        # Getting optimal parameters
        self.best_score = grid.best_score_   
        self.best_params= grid.best_params_
        self.best_model = grid.best_estimator_        

        
    def model_evaluation(self):
        # Getting train and test accuracy        
        self.best_model.fit(self.X_train, self.y_train)
        self.y_pred = self.best_model.predict(self.X_test)
        self.train_accuracy = self.best_model.score(self.X_train, self.y_train)
        self.test_accuracy = self.best_model.score(self.X_test, self.y_test)
        
        
    def k_fold_cross_validation(self):
        # Model definition
        if self.scaler:
            pipeline = make_pipeline(StandardScaler(),self.best_model)
        else:
            pipeline = self.best_model
        # Applying k-fold and getting scores
        self.kfold_scores = cross_val_score(pipeline, self.X, self.y, cv=self.k_folds)
        
        
    def markdown_report(self):
        display(Markdown(
        f"""---
        \n Model: {self.model_name}
        \nDataset size {len(self.X)}, Test size {self.test_size*100:.1f}% 
        \n*Optimal parameters (GridSearchCV {self.k_folds}-Folds ):*
        \n   - Best params **{self.best_params}**
        \n   - Accuracy **{self.best_score:.2f}** 
        \n*Model_evaluation (test size {self.test_size*100:.1f}%):*
        \n   -  Train accuracy **{self.train_accuracy:.2f}**
        \n - Test accuracy **{self.test_accuracy:.2f}**
        \n*Cross Validation Score :*
        \n   - Avarage Accuracy **{np.mean(self.kfold_scores):.2f}**   +/-{np.std(self.kfold_scores):.2f} \n---""" ))
        
        # Plotting the confusion matrix and a report to see the main scores of our model
        sns.set(style="white")
        plot_confusion_matrix(self.best_model, self.X_test, self.y_test) 
        plt.show() 
   
        print('='*20,'Training Set Results','='*20)
        print(classification_report(self.y_train, self.best_model.predict(self.X_train)))

        print('='*20,'Testing Set Results','='*20)
        report_testing_dtree = classification_report(self.y_test, self.y_pred)
        print(report_testing_dtree)
        print('='*60)
