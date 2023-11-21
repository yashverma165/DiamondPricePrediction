import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import os,sys


@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifact','model.pkl')
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variabloes from train and test')
            X_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                'linear Regression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'Decision Tree':DecisionTreeRegressor()
            }
            
            
            model_report:dict=evaluate_model(X_train,y_train,x_test,y_test,models)
            print(model_report)
            print('\n=========================================================================')
            logging.info(f"Model Report :{model_report}") 
            
            # To get best model scor from dictionary
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            print(f"Best Model Found , Model name : {best_model_name} , R2 Score : {best_model_score}")
            print('\n===============================================================================')
            logging.info(f"Best Model Found , Model name : {best_model_name} , R2 Score : {best_model_score}")
            
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.info('Exception Occured at Model Training')
            raise CustomException(e,sys)    