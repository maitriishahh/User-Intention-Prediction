import os
import sys
import joblib
import pickle
import numpy as np
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

            self.scaler_path = os.path.join(
                "artifacts",
                "scaler.pkl"
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
            logging.info("Loading transformed test data")

            with open(self.transformed_data_path, "rb") as file:
                X_train, X_test, y_train, y_test = pickle.load(file)

            logging.info(f"Test data shape: {X_test.shape}")

            return X_test.sample(5)

        except Exception as e:
            raise AppException(e, sys)
        
    def prepare_input(self, user_input):
        feature_columns_path = os.path.join(
            "artifacts",
            "dataset",
            "transformed_data",
            "feature_columns.pkl"
        )

        with open(feature_columns_path, "rb") as f:
            feature_columns = pickle.load(f)

        user_df = pd.DataFrame([user_input])

        user_df = pd.get_dummies(user_df)

        user_df = user_df.reindex(columns=feature_columns, fill_value=0)
        user_df = user_df.astype(float)

        logging.info(f"Prepared input:\n{user_df}")

        return user_df


    def predict(self, model, data):
        try:
            logging.info("Making predictions")

            # Prepare UI input
            if isinstance(data, dict):
                data = self.prepare_input(data)

            # Load scaler
            scaler = joblib.load(self.scaler_path)
            logging.info("Scaler loaded successfully")

            # Convert to numpy if dataframe
            if isinstance(data, pd.DataFrame):
                data = data.values

            # Apply scaling
            data = scaler.transform(data)

            # Predict probabilities
            probabilities = model.predict_proba(data)[:, 1]

            threshold = 0.4
            logging.info(f"Using Threshold: {threshold}")

            prediction = (probabilities >= threshold).astype(int)

            labels = ["No Purchase" if p == 0 else "Purchase" for p in prediction]

            logging.info(f"Prediction probabilities: {probabilities}")
            logging.info(f"Predictions: {prediction}")
            logging.info(f"Prediction labels: {labels}")

            output = pd.DataFrame({
                "Probability": probabilities,
                "Prediction": prediction,
                "Label": labels
            })

            output_path = os.path.join("artifacts", "predictions.csv")
            output.to_csv(output_path, index=False)

            logging.info(f"Predictions saved at {output_path}")

            return prediction, probabilities

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