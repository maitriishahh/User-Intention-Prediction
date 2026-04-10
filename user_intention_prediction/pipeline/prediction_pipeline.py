import os
import sys
import joblib
import pickle
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

            self.transformed_data_path = os.path.join(
                "artifacts",
                "dataset",
                "transformed_data",
                "transformed_data.pkl"
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


    def load_test_data(self):
        try:
            logging.info("Loading test data")

            with open(self.transformed_data_path, "rb") as file:
                X_train, X_test, y_train, y_test = pickle.load(file)

            logging.info(f"Test data loaded successfully with shape {X_test.shape}")

            return X_test.head(5)

        except Exception as e:
            raise AppException(e, sys)


    def predict(self, model, data):
        try:
            logging.info("Making predictions")

            prediction = model.predict(data)

            logging.info("Prediction completed")

            return prediction

        except Exception as e:
            raise AppException(e, sys)


    def start_prediction_pipeline(self):
        try:
            model = self.load_model()

            test_data = self.load_test_data()

            prediction = self.predict(model, test_data)

            logging.info(f"{'='*20}Prediction Pipeline Completed{'='*20}")

            return prediction

        except Exception as e:
            raise AppException(e, sys)