import sys
import pandas as pd
from src.exception import Custom_Exception
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        """
        Initialize PredictPipeline class with configurations.
        """
        pass
    
    def predict(self, features):
        """
        Method to predict the data point.        
        """
        try:
            model_path = "artifacts/trained_model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise Custom_Exception(e, sys)
    
class CustomData():
    
    def __init__(self, gender:str, race_ethnicity:str, parental_level_of_education, 
                 lunch:str, test_preparation_course:str, reading_score:int, writing_score:int):
        """
        Initialize CustomData class with configurations.
        """
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    
    def get_data_as_df(self):
        """
        Method to get data as dataframe.
        """
        try:
            custom_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            
            return pd.DataFrame(custom_data_dict)
        
        except Exception as e:
            raise Custom_Exception(e, sys)
        
    