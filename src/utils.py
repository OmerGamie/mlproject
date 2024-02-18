import os
import sys

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import Custom_Exception


def save_object(file_path, obj):
    """
    Method to save the object to the file path.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise Custom_Exception(e, sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models):
    """
    Method to evaluate the models
    """
    try:
        # Dictionary to store the model report
        report = {}
        
        # Looping through the models
        for m in range(len(list(models))):
            model = list(models.values())[m] # Getting the model
            
            model.fit(x_train, y_train) 
            
            # Predicting the values
            y_train_pred = model.predict(x_train)
            
            y_test_pred = model.predict(x_test)
            
            # Calculating the r2 score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Storing the model report
            report[list(models.keys())[m]] = test_model_score
            
        return report
   
    except Exception as e:
        raise Custom_Exception(e, sys)
            
            