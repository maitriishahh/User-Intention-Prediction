# User Intention Prediction

## Workflow:
Update:
- config.yaml
- entity/config_entity.py
- config/configuration.py
- components
- pipeline/training_pipeline.py
- main.py
- app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/maitriishahh/User-Intention-Prediction.git
```
### STEP 01 - Create a conda environment after opening the repository

```bash
conda create -n venv python==3.13.5 -y
```

```bash
conda activate venv
```
### STEP 02 - Install the requirements
```bash
pip install -r requirements.txt
```

Now run,
```bash
stramlit run app.py
```