import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],#take out last columns
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
               "Linear Regression": LinearRegression(),
               "K-Neighbors Classifier": KNeighborsRegressor(),
               "Decision Tree": DecisionTreeRegressor(),
               "Random Forest": RandomForestRegressor(),
               "XGBRegressor": XGBRegressor(), 
               "Gradient Boosting":GradientBoostingRegressor(),
               "CatBoosting Regressor": CatBoostRegressor(verbose=False),
               "AdaBoost Classifier": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)
            
            #To get best model score from dict
            best_model_score=max(sorted(model_report.values()))
            #To get best model form dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score) #using list
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model#the pickle file want to create
            )  
            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predicted)

            print(f"R² Score on Test Data = {r2_square:.4f}")

            return r2_square


        except Exception as e:
            raise CustomException(e,sys)