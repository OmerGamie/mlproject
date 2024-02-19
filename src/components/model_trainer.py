import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import Custom_Exception
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """
    Dataclass to store the configuration of the model trainer
    """
    trained_model_path=os.path.join("artifacts", "trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        """
        Constructor to initialize the ModelTrainer class
        """
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        Method to initiate the model training process
        """
        try:
            logging.info("Splitting train and test input data")
            
            # Splitting the input data into train and test
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            
            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            # Dictionary to store the models
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
            
            # Dictionary to store the model parameters
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            # Evaluating the models and getting the model report
            model_report:dict =evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, 
                                               models=models, param=params)
        
            # Getting the best model name and score
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            # Fitting the best model
            if best_model_score < 0.6:
                raise Custom_Exception("Best model score is not found")
            
            # Saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )    
            
            logging.info("Best model score found and saved successfully")
            
            # Returning the best model score
            prediction = best_model.predict(x_test)
            r2_score_value = r2_score(y_test, prediction)
            return r2_score_value
            
        except Custom_Exception as e:
            raise Custom_Exception(e, sys)