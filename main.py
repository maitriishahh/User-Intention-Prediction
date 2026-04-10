from user_intention_prediction.pipeline.training_pipeline import TrainingPipeline
from user_intention_prediction.pipeline.prediction_pipeline import PredictionPipeline
from user_intention_prediction.logger.log import logging


if __name__ == "__main__":
    
    logging.info("Starting Training Pipeline")
    
    training = TrainingPipeline()
    training.start_training_pipeline()

    logging.info("Starting Prediction Pipeline")

    prediction = PredictionPipeline()
    result = prediction.initiate_prediction("sample.csv")

    print("Prediction:", result)