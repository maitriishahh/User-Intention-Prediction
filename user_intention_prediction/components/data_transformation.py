import os
import sys
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from user_intention_prediction.logger.log import logging
from user_intention_prediction.exception.exception_handler import AppException
from user_intention_prediction.config.configuration import AppConfiguration


class DataTransformation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20}")
            self.data_transformation_config = app_config.get_data_transformation_config()
        except Exception as e:
            raise AppException(e,sys) from e


    def transform_data(self):
        try:
            clean_data_path = self.data_transformation_config.clean_data_file_path

            df = pd.read_csv(clean_data_path)

            logging.info(f"Dataset shape before transformation: {df.shape}")

            # Drop missing values (based on notebook)
            df = df.dropna()

            logging.info(f"Dataset shape after dropping missing values: {df.shape}")

            # One hot encoding (automatic - based on notebook)
            df = pd.get_dummies(df, drop_first=True)

            logging.info("Categorical encoding completed")

            # Target variable
            X = df.drop("Revenue", axis=1)
            y = df["Revenue"]

            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42, stratify=y
            )

            logging.info("Train test split completed")

            transformed_data_dir = self.data_transformation_config.transformed_data_dir
            os.makedirs(transformed_data_dir, exist_ok=True)

            transformed_data_file_path = os.path.join(
                transformed_data_dir,
                "transformed_data.pkl"
            )

            # Save transformed data
            with open(transformed_data_file_path, "wb") as file:
                pickle.dump(
                    (X_train, X_test, y_train, y_test),
                    file
                )

            logging.info(f"Transformed data saved at {transformed_data_file_path}")

        except Exception as e:
            raise AppException(e,sys) from e


    def initiate_data_transformation(self):
        try:
            self.transform_data()

            logging.info(f"{'='*20}Data Transformation log completed.{'='*20}\n\n")

        except Exception as e:
            raise AppException(e,sys) from e