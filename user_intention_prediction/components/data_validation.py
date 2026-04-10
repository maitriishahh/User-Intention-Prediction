import os
import sys
import pandas as pd

from user_intention_prediction.logger.log import logging
from user_intention_prediction.exception.exception_handler import AppException
from user_intention_prediction.config.configuration import AppConfiguration


class DataValidation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20}")
            self.data_validation_config = app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e,sys) from e


    def validate_dataset(self):
        try:
            data_file_path = self.data_validation_config.data_csv_file

            if not os.path.exists(data_file_path):
                raise Exception(f"Dataset not found at {data_file_path}")

            logging.info("Dataset found successfully")

            df = pd.read_csv(data_file_path)

            logging.info(f"Dataset shape before cleaning: {df.shape}")
            #Remove blank rows
            df = df.dropna()
            
            # Missing values
            missing_values = df.isnull().sum()
            logging.info(f"Missing values:\n{missing_values}")

            # Duplicate check
            duplicate_count = df.duplicated().sum()
            logging.info(f"Duplicate rows: {duplicate_count}")

            # Remove duplicates
            df = df.drop_duplicates().reset_index(drop=True)

            logging.info(f"Dataset shape after removing duplicates: {df.shape}")

            # Save clean data
            clean_data_dir = self.data_validation_config.clean_data_dir
            os.makedirs(clean_data_dir, exist_ok=True)

            clean_data_file_path = os.path.join(
                clean_data_dir,
                "clean_data.csv"
            )

            df.to_csv(clean_data_file_path, index=False)

            logging.info(f"Clean data saved at {clean_data_file_path}")

        except Exception as e:
            raise AppException(e,sys) from e


    def initiate_data_validation(self):
        try:
            self.validate_dataset()

            logging.info(f"{'='*20}Data Validation log completed.{'='*20}\n\n")

        except Exception as e:
            raise AppException(e,sys) from e