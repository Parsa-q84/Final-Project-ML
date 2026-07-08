# End-to-End MLOps Customer Churn Prediction Pipeline

This repository contains a structured, end-to-end MLOps pipeline designed to predict customer churn using the **IBM Telco Customer Churn** dataset. The project covers data versioning, feature engineering, multi-model tracking via MLflow, and automated containerized deployment using Docker.

---

## Project Structure
The project strictly follows the standard machine learning operations directory architecture:

├── data/
│   ├── v1/          # Raw dataset version (unmodified)
│   ├── v2/          # Cleaned dataset (missing values handled & encoded)
│   └── v3/          # Feature engineered dataset (scaled and new features added)
├── src/
│   ├── data_loader.py    # Ingests and versions raw data to v1
│   ├── preprocessing.py  # Cleans data and encodes categorical features to v2
│   ├── features.py       # Creates interactions and scales numerical values to v3
│   ├── mlflow_utils.py   # Configures MLflow logging and experiment tracking
│   ├── train.py          # Trains multiple models (XGBoost, LightGBM, CatBoost, RF) with 5-Fold CV
│   ├── evaluate.py       # Computes final test metrics and logs confusion matrix artifacts
│   └── run_pipeline.py   # Main orchestrator to run the complete workflow automatically
├── Dockerfile            # Container configuration for serving the final model
└── README.md             # Project documentation

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ and Docker installed on your system.

### 2. Installation
Install the required dependencies inside your virtual environment:

pip install pandas numpy scikit-learn xgboost lightgbm catboost mlflow matplotlib seaborn openpyxl skops
3. Launch MLflow Server
Open a separate terminal window and start the local MLflow tracking server:

Bash
mlflow server --host 127.0.0.1 --port 5000
The MLflow dashboard will be accessible at http://127.0.0.1:5000.

4. Execute the End-to-End Pipeline
Run the main pipeline orchestrator to automatically process data, train all 12 model-version combinations, and log artifacts:

Bash
python src/run_pipeline.py
Containerized Deployment (Docker)
The best-performing model trained on the latest dataset version is packaged and served as a REST API using Docker.

Build the Docker Image:
Bash
docker build -t churn-model-service:v1 .
Run the Container:
Bash
docker run -d -p 5001:5001 --name churn_container churn-model-service:v1
The prediction service will now be active on port 5001, ready to accept payload inference requests.
