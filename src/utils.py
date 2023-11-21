import os,sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
 
 
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
     
    except Exception as e:
        raise CustomException(e, sys)    
    
 
def evaluate_model(X_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            
            # Train model
            model.fit(X_train,y_train)
            
            #Predict Testing Data
            y_pred=model.predict(x_test)
            
            # Get r2 score for the train and test data
            test_model_score=r2_score(y_test,y_pred)
            
            report[list(models.keys())[i]] = test_model_score
         
        return report 
    
    except Exception as e:
        logging.info('Exception Occured during model training ')
        raise CustomException(e,sys)    
    