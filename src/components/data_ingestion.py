# import os
# import sys
# from src.exception import CustomException
# from src.logger import logging

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass #used to create class variables

# @dataclass
# # this decorator is used when you have to do basic things like giving a file path or so.
# # but, if you have to write some functions that do some tasks, it is preferable to continue with __init__
# class DataIngestionConfig:
#     raw_data_path:str = os.path.join('artifacts','raw_data.csv')
#     train_data_path: str= os.path.join('artifacts','train_data.csv')
#     test_data_path: str=os.path.join('artifacts','test_data.csv')
#     # we created 3 files namely raw_data_path, train_data_path and test_data_path

# class DataIngestion:
#     # data ingestion class. here we do train_test_split and save training and testing data to files 
#     # mentioned above.
#     def __init__(self):
#         self.ingestion_config = DataIngestionConfig()
#         #all files are stored in the variable 'ingestion_config'

#     def initiate_data_ingestion(self):
#         logging.info('started to ingest data')
#         # we keep on doing logging to keep track of what exaclty we are doing (consider it as comments)
#         try:
#             data = pd.read_csv('notebooks\student.csv')
#             # we read raw data
#             logging.info('read the data')
#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
#             # we made one directory
#             data.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
#             # saved raw data to raw_data_path

#             train_data, test_data = train_test_split(data, random_state=42)
#             # after splitting raw data, we saved train set to train_data_path and test set to test_data_path
#             train_data.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
#             test_data.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
#             logging.info('training and testing data separated')

#             return(
#                 # we are basically returning these files to use them further in data_preprocessing
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path

#             )


#         except Exception as e:
#             raise CustomException
        

# if __name__=='__main__':
#     obj=DataIngestion()
#     train_data,test_data=obj.initiate_data_ingestion()
#     # obj.initiate_data_ingestion()



import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_preprocessing import DataTransformation
from src.components.data_preprocessing import DataTransformationConfig

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    # it provides files that may be necessary for this particular python file
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebooks\student.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

