import os 
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

## Initialize the data ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path=os.path.join('artifact','train.csv')
    test_data_path=os.path.join('artifact','test.csv')
    raw_data_path=os.path.join('artifact','raw.csv')



## Create data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
        
        
    def initiate_data_ingestion(self):
        logging.info('Data inesgtion method starts')
        
        try:
            df=pd.read_csv(os.path.join('notebooks\data','gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')
            
            ## If artifacts folder does not exist we can make aritfacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            ## because data is raw we need save to csv 
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info('Train Test Split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            
            logging.info('split done')
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion of data is completed')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )   
            
        except Exception as e:
            logging.info('Error Occured in Data Inegistion Config')    
            
     
   