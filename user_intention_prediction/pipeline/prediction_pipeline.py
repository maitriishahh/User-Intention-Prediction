import os
import sys
import joblib
import pandas as pd

from user_intention_prediction.logger.log import logging
from user_intention_prediction.exception.exception_handler import AppException


class PredictionPipeline:

    def __init__(self):
        try:
            logging.info(f"{'='*20}Prediction Pipeline Started{'='*20}")
            
            self.model_path = os.path.join(
                "artifacts",
                "trained_model",
                "model.pkl"
            )

        except Exception as e:
            raise AppException(e, sys)


    def load_model(self):
        try:
            logging.info("Loading trained model")

            model = joblib.load(self.model_path)

            logging.info("Model loaded successfully")

            return model

        except Exception as e:
            raise AppException(e, sys)


    def load_data(self, data_path):
        try:
            logging.info(f"Loading data from {data_path}")

            data = pd.read_csv(data_path)

            logging.info(f"Data loaded successfully with shape {data.shape}")

            return data

        except Exception as e:
            raise AppException(e, sys)


    def predict(self, data):
        try:
            model = self.load_model()

            logging.info("Making predictions")

            prediction = model.predict(data)

            logging.info("Prediction completed")

            return prediction

        except Exception as e:
            raise AppException(e, sys)


    def initiate_prediction(self, data_path):
        try:
            data = self.load_data(data_path)

            prediction = self.predict(data)

            logging.info(f"{'='*20}Prediction Pipeline Completed{'='*20}")

            return prediction

        except Exception as e:
            raise AppException(e, sys)