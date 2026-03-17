import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn as mlflow_sklearn
import mlflow.xgboost as mlflow_xgboost
from mlflow.models.signature import infer_signature
from pydantic import BaseModel
from logging import Logger
from dotenv import load_dotenv

from sklearn.model_selection import cross_validate, StratifiedKFold

from src.helpers.initiate_stage import initiate_file
from src.helpers.exception_handling import MyException
from src.helpers.load_save import save_artifacts
from src.storage.minio_client import MinIOClient
from src.helpers.model_factory import ModelFactory

import warnings
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)

load_dotenv()

class TrainingConfig(BaseModel):
    bucket_name: str
    data_path: str
    selected_model: str
    n_trials: int
    optuna_scoring: str
    experiment_name: str
    output_paths: dict

class ModelTraining:
    def __init__(self, config: TrainingConfig, logger: Logger):
        self.config = config
        self.logger = logger
        
        try:
            self.minio_client = MinIOClient(
                endpoint_url=os.getenv("END_POINT_URL", ''), 
                access_key=os.getenv('ACCESS_KEY', ''), 
                secret_key=os.getenv('SECRET_KEY', ''), 
                bucket_name=self.config.bucket_name
            )
        except Exception as e:
            raise MyException(e, sys, self.logger)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ''))
        mlflow.set_experiment(self.config.experiment_name)

    def load_transformed_data(self):
        try:
            data = np.load(self.config.data_path)
            self.logger.info("Transformed training data loaded successfully.")
            return data['X'], data['y']
        except Exception as e:
            raise MyException(e, sys, self.logger)

    def optimize_hyperparameters(self, X_train, y_train):
        self.logger.info(f"Starting Optuna Optimization for {self.config.selected_model}")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scoring_metrics = {
            'f1': 'f1', 
            'precision': 'precision', 
            'recall': 'recall', 
            'roc_auc': 'roc_auc'
        }

        def objective(trial):
            params = ModelFactory.get_optuna_space(trial, self.config.selected_model)
            model = ModelFactory.get_model(self.config.selected_model, y_train, **params)
            
            cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring_metrics)
            
            f1_mean = cv_results['test_f1'].mean()
            precision_mean = cv_results['test_precision'].mean()
            recall_mean = cv_results['test_recall'].mean()
            roc_auc_mean = cv_results['test_roc_auc'].mean()

            with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "f1_score": f1_mean,
                    "cv_precision": precision_mean,
                    "cv_recall": recall_mean,
                    "cv_roc_auc": roc_auc_mean
                })

            return f1_mean 

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials)
        
        self.logger.info(f"Optuna Search Complete. Best F1-Score: {study.best_value}")
        return study

    def train_final_model(self, X_train, y_train, best_params, best_score):
        self.logger.info("Training final Champion model with optimized parameters.")
        
        mlflow.log_params(best_params)
        mlflow.log_param("Algorithm", self.config.selected_model)
        mlflow.log_metric("cv_best_f1_score", best_score)
        
        final_model = ModelFactory.get_model(self.config.selected_model, y_train, **best_params)
        final_model.fit(X_train, y_train)
        
        predictions = final_model.predict(X_train)
        signature = infer_signature(X_train, predictions)

        pip_reqs = [
            "scikit-learn",
            "skops",
            "xgboost",
            "numpy",
            "pandas"
        ]       
        if self.config.selected_model == "XGBoost":
            mlflow_xgboost.log_model(
                final_model, 
                name="model", 
                signature=signature,
                pip_requirements=pip_reqs
            )
        else:
            mlflow_sklearn.log_model(
                final_model, 
                name="model", 
                signature=signature,
                serialization_format="skops",
                pip_requirements=pip_reqs
            )    
        return final_model

    def save_and_upload(self, final_model, study):
        try:
            model_path = self.config.output_paths["model"]
            save_artifacts(final_model, model_path, self.logger)
            
            report_path = self.config.output_paths["optuna_report"]
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            report_data = {
                "algorithm": self.config.selected_model,
                "best_score": study.best_value,
                "best_params": study.best_params
            }
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=4)
                
            self.logger.info("Artifacts saved locally.")

            self.minio_client.upload_file(model_path, "artifacts/models/trained_model.joblib")
            self.minio_client.upload_file(report_path, "artifacts/reports/optuna_study.json")
            self.logger.info("Artifacts synced to MinIO remote successfully.")
        except Exception as e:
            raise MyException(e, sys, self.logger)

    def run(self):
        while mlflow.active_run():
            mlflow.end_run()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        parent_run_name = f"{self.config.selected_model}[{timestamp}]"

        try:
            X_train, y_train = self.load_transformed_data()
            
            with mlflow.start_run(run_name=parent_run_name):
                study = self.optimize_hyperparameters(X_train, y_train)
                
                final_model = self.train_final_model(
                    X_train, 
                    y_train, 
                    study.best_params, 
                    study.best_value
                )
            self.save_and_upload(final_model, study)
            self.logger.info("Model Training Stage Completed Successfully")
            
        except Exception as e:
            while mlflow.active_run():
                mlflow.end_run()
            raise MyException(e, sys, self.logger)

if __name__ == "__main__":
    params, logger = initiate_file(
        params_file_path="config/training.yaml",
        component_name="ModelTraining",
    )
    config = TrainingConfig(**params)
    ModelTraining(config, logger).run()