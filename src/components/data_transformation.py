import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # input file path
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Main method to perform data transformation operations.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        """
        Method to get the transformer object.
        """
        try:
            numerical_features = ['writing score', 'reading score']
            categorical_features = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]
            
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler()),
            ]) 
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])
            
            logging.info("Numerical and Categorical pipelines created successfully.")
            
            # Create a preprocessor pipeline
            preprocess_pipeline = ColumnTransformer(
                [
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ])
            return preprocess_pipeline
        
        except Exception as e:
            raise Custom_Exception(e, sys)
        
    
    def initiate_transformation(self, train_data_path, test_data_path):
        """
        Method to initiate data transformation process.
        """
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            
            logging.info("Data loaded successfully.")
            
            # Create a preprocessor pipeline
            logging.info("Obtaining data transformation object...")
            
            preprocessing_obj = self.get_data_transformer_obj()
            
            # Define target column and input features
            target_column = "math score"
            numerical_features = ["writing score", "reading score"]
            
            # Split the data into input features and target column
            input_features_train =train_data.drop(columns=[target_column], axis=1)
            target_column_train = train_data[target_column]
            
            input_features_test = test_data.drop(columns=[target_column], axis=1)
            target_column_test = test_data[target_column]
            
            logging.info("Applying preprocessing object to train and test data...")
            
             
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessing_obj.transform(input_features_test)
            
            train_arr = np.c_[input_features_train_arr, np.array(target_column_train)]
            test_arr = np.c_[input_features_test_arr, np.array(target_column_test)]
            
            logging.info("Saving preprocessor object...")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        
        except Exception as e:
            raise Custom_Exception(e, sys)
            
            
            

        

            
        