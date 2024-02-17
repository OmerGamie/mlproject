import os
import sys
import pandas as pd 
from src.exception import Custom_Exception
from src.logger import logging
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "tain.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        """
        Initialize DataIngestion class with configurations.
        """
        self.ingestion_config = DataIngestionConfig() # Configurations for data sources
    
    
    def initiate_data_ingestion(self):
        """
        main method to intiate data ingestion process.
        """
        logging.info("Intiating data ingestion process...")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Data loaded successfully.")
            
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
         
                
            # Split data into train and test sets
            logging.info("Splitting data into train and test sets...")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
                
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            test_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            logging.info("Data ingestion process completed successfully") 
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            # Log and raise custom exception if any error occurs
            raise Custom_Exception(e, sys)
            

if __name__ == "__main__":
    # Example usage of DataIngestion class
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    

                
                
                
            
        



