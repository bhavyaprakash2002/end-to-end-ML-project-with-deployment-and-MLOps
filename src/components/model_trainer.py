import os
import sys
import dill
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, model_evaluate

@dataclass
class model_trainer_config:
    model_obj_file_path = os.path.join('artifacts','model.pkl')

class model_training:
    def __init__(self):
        self.model_config = model_trainer_config()

    def initiate_model_trainer(self, train_arr, test_arr):
        logging.info('model training started')
        try:
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            ) 

            models = {
                'linear_regression' : LinearRegression(),
                'KNN' : KNeighborsRegressor(),
                'Decision_tree' : DecisionTreeRegressor(),
                'random_forest' : RandomForestRegressor(),
                'xgboost' : XGBRegressor()
                
            }

            model_report:dict = model_evaluate(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, models = models)
            best_model_score = min(model_report.values())
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            save_object(
                file_path=self.model_config.model_obj_file_path,
                obj = best_model
            )
            predictions = best_model.predict(x_test)
            rmse = mean_squared_error(y_test, predictions)
            return rmse

        except Exception as e:
            raise CustomException(e, sys)







