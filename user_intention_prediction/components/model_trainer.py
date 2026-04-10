import os
import sys
import pickle
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

from user_intention_prediction.logger.log import logging
from user_intention_prediction.exception.exception_handler import AppException
from user_intention_prediction.config.configuration import AppConfiguration


class ModelTrainer:

    def __init__(self, app_config = AppConfiguration()):
        try:
            logging.info(f"{'='*20}Model Trainer log started.{'='*20}")
            self.model_trainer_config = app_config.get_model_trainer_config()
        except Exception as e:
            raise AppException(e, sys)


    def tune_model(self, model, param_grid, X_train, y_train, X_test, y_test):

        try:
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
                scoring='roc_auc'
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logging.info(f"{model.__class__.__name__} Best Params: {grid.best_params_}")
            logging.info(f"{model.__class__.__name__} Accuracy: {accuracy}")
            logging.info(f"{model.__class__.__name__} ROC AUC: {roc_auc}")

            return best_model, roc_auc

        except Exception as e:
            raise AppException(e, sys)


    def train_model(self):

        try:
            transformed_data_file = self.model_trainer_config.transformed_data_file_dir

            with open(transformed_data_file, "rb") as file:
                X_train, X_test, y_train, y_test = pickle.load(file)

            logging.info("Data loaded successfully")

            # Scaling
            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            logging.info("Feature scaling completed")

            # Hyperparameters

            logistic_params = {
                'C': [0.01, 0.1, 1, 10]
            }

            decision_tree_params = {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }

            random_forest_params = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10]
            }

            # Train models

            lr_model, lr_score = self.tune_model(
                LogisticRegression(max_iter=1000, class_weight='balanced'),
                logistic_params,
                X_train, y_train, X_test, y_test
            )

            dt_model, dt_score = self.tune_model(
                DecisionTreeClassifier(class_weight='balanced'),
                decision_tree_params,
                X_train, y_train, X_test, y_test
            )

            rf_model, rf_score = self.tune_model(
                RandomForestClassifier(class_weight='balanced'),
                random_forest_params,
                X_train, y_train, X_test, y_test
            )

            models = {
                "Logistic Regression": (lr_model, lr_score),
                "Decision Tree": (dt_model, dt_score),
                "Random Forest": (rf_model, rf_score)
            }

            # Select best model

            best_model_name = None
            best_score = 0
            best_model = None

            for model_name, (model, score) in models.items():

                logging.info(f"{model_name} ROC AUC Score: {score}")

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = model_name

            logging.info(f"Best Model Selected: {best_model_name}")

            trained_model_dir = self.model_trainer_config.trained_model_dir

            os.makedirs(trained_model_dir, exist_ok=True)

            trained_model_path = os.path.join(
                trained_model_dir,
                self.model_trainer_config.trained_model_name
            )

            joblib.dump(best_model, trained_model_path)

            logging.info(f"Best model saved at {trained_model_path}")

        except Exception as e:
            raise AppException(e, sys)


    def initiate_model_trainer(self):

        try:
            self.train_model()

            logging.info(f"{'='*20}Model Trainer log completed.{'='*20}\n\n")

        except Exception as e:
            raise AppException(e, sys)