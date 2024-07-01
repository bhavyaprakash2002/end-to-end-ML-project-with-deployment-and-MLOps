import os
import sys
from dataclasses import dataclass
from src.utils import save_object

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def data_transformation(self):
        # all transformation/preprocessing will happen here
        logging.info('transformation started')
        try:
            num_features = ['reading_score','writing_score']
            cat_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline([('scaling', StandardScaler(with_mean=False))])
            cat_pipeline = Pipeline([('encoding', OneHotEncoder(handle_unknown='ignore')),
                                     ('scaling_cat_features', StandardScaler(with_mean=False))])
            
            
            transformation = ColumnTransformer([('num_pipeline', num_pipeline, num_features),
                                                ('cat_pipeline', cat_pipeline, cat_features)])
            
            logging.info('transformation of numerical and categorical features done!')
            
            return transformation

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info('data transformation initialized')
            transformation_obj = self.data_transformation()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            train_data_input_features = train_data.drop(columns=[target_column_name], axis = 1)
            test_data_input_features = test_data.drop(columns=[target_column_name], axis = 1)

            train_data_target = train_data['math_score']
            test_data_target = test_data['math_score']

            train_arr_input_features = transformation_obj.fit_transform(train_data_input_features)
            test_arr_input_features = transformation_obj.transform(test_data_input_features)

            train_arr = np.c_[train_arr_input_features, np.array(train_data_target)]
            test_arr = np.c_[test_arr_input_features, np.array(test_data_target)]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = transformation_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
            
            

        except Exception as e:
            raise CustomException(e,sys)
