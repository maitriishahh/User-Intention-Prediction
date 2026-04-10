from user_intention_prediction.components.data_ingestion import DataIngestion
from user_intention_prediction.components.data_validation import DataValidation
from user_intention_prediction.components.data_transformation import DataTransformation
from user_intention_prediction.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training_pipeline(self):
        """starts the training pipeline 
        :return: none"""
        self.data_ingestion.initiate_data_ingestion()
        self.data_validation.initiate_data_validation()
        self.data_transformation.initiate_data_transformation()
        self.model_trainer.initiate_model_trainer()