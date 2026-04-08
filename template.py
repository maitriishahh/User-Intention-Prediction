import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "user_intention_prediction"

list_of_files = [
    # Project package
    f"{project_name}/__init__.py",

    # Components
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",

    # Config
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",

    # Constants
    f"{project_name}/constant/__init__.py",

    # Entity
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",

    # Exception
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception_handler.py",

    # Logger
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/log.py",

    # Pipeline
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",

    # Utils
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/util.py",

    # Root level files
    "app.py",
    "setup.py",
    "requirements.txt",
    "README.md",

    # Config YAML
    "config/config.yaml",

    # Extra folders
    "artifacts/.gitkeep",
    "logs/.gitkeep",
    "data/.gitkeep",
    "notebooks/.gitkeep",
    "models/.gitkeep"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")