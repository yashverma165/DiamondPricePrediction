from sklearn.impute import SimpleImputer  ## for Handaling misssing values
from sklearn.preprocessing import StandardScaler ## Feature scaleing
from sklearn.preprocessing import OrdinalEncoder #Ordinal Encoding

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

import sys,os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


## Data Transformation config

@dataclass
class DataTransformatiomconfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')
    



## Data Ingestion Config Class    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformatiomconfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should bo ordinal-encoded amd which should be scaled
            categorical_col=['cut','color','clarity']
            numerical_col=['carat','depth','table','x','y','z']
            
            
            # Define Custon Rankaing for each ordianl variable
            cut_col=['Fair', 'Good', 'Very Good','Premium','Ideal']
            clarity_col=['I1','SI2',"SI1",'VS2','VS1','VVS2','VVS1','IF']
            color_col=['D','E','F','G','H','I','J']
            
            logging.info('Pipeline Initiated')
            
            # Numerical pipline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            # Categorical Pipline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_col,color_col,clarity_col])),
                    ('standardscaler',StandardScaler())
                    
                ]
            )
            
            preprocessor=ColumnTransformer([
                ('num_pipline',num_pipeline,numerical_col),
                ('cat_pipline',cat_pipeline,categorical_col)
            ])
            
            return preprocessor
        
            logging.info('Pipline Completed')
            
        except Exception as e:
            
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
       
       
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading Train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('Read train and test data compeleted')
            logging.info(f'Train Dataframe Head : \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n {test_df.head().to_string()}')
            
            logging.info('Obtaining Preprocessor object')
            
            preprocessing_obj=self.get_data_transformation_object()
            
            target_col= 'price'
            drop_columns = [target_col,'Unnamed: 0']
            
            # features into independet and dependent features
            input_features_train_df=train_df.drop(columns=drop_columns,axis=1)
            traget_feature_train_df=train_df[target_col]
            
            input_features_test_df=test_df.drop(columns=drop_columns,axis=1)
            traget_feature_test_df=test_df[target_col]
            
            # Apply transformation
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)
            
            logging.info('Appling preprocessor object on training and testing datatset')
            
            train_arr = np.c_[input_feature_train_arr,np.array(traget_feature_train_df)]
            test_arr  = np.c_[input_feature_test_arr,np.array(traget_feature_test_df)]
            
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info(' Preprocessor file is created and saved')
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
            
               
        except Exception as e:
            logging.info('Exception occured in the initiate Datatran sformation')
            raise CustomException(e,sys)
                         