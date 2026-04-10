import os
import sys
import urllib.request as urllib
import shutil
from user_intention_prediction.logger.log import logging
from user_intention_prediction.exception.exception_handler import AppException
from user_intention_prediction.config.configuration import AppConfiguration


class DataIngestion:
    def __init__(self, app_config = AppConfiguration()):
        """data ingestion initialization"""
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20}")
            self.data_ingestion_config = app_config.get_data_ingestion_config()
        except Exception as e:
            raise AppException(e,sys) from e
        
    def download_data(self):
        """Fetch the data from the url"""
        try:
            dataset_url = self.data_ingestion_config.dataset_download_url
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            os.makedirs(raw_data_dir, exist_ok=True)

            data_file_name = os.path.basename(dataset_url)
            data_file_path = os.path.join(raw_data_dir, data_file_name)

            logging.info(f"Downloading data from {dataset_url} into file {data_file_path}")
            urllib.urlretrieve(dataset_url, data_file_path)
            logging.info(f"Downloaded data from {dataset_url} into file {data_file_path}")

            return data_file_path
        
        except Exception as e:
            raise AppException(e,sys) from e
        
    def initiate_data_ingestion(self):
        try:
            raw_file_path = self.download_data()

            ingested_dir = self.data_ingestion_config.ingested_dir
            os.makedirs(ingested_dir, exist_ok=True)

            file_name = os.path.basename(raw_file_path)
            ingested_file_path = os.path.join(ingested_dir, file_name)

    
            shutil.copy(raw_file_path, ingested_file_path)

            logging.info(f"Copied file to {ingested_file_path}")
            logging.info(f"{'='*20}Data Ingestion log completed.{'='*20}\n\n")

        except Exception as e:
            raise AppException(e,sys) from e