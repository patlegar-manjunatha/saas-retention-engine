import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class ModelFactory:
    """
    Factory pattern to dynamically instantiate models and their Optuna search spaces
    while standardizing parameters like n_jobs and class balancing.
    """
    
    @staticmethod
    def get_model(model_name: str, y_train: np.ndarray, **kwargs):
        
        neg_class = len(y_train[y_train == 0])
        pos_class = len(y_train[y_train == 1])
        scale_pos_weight = neg_class / pos_class if pos_class > 0 else 1.0

        if model_name == "RandomForest":
            return RandomForestClassifier(class_weight="balanced", n_jobs=-1, **kwargs)
        
        elif model_name == "LogisticRegression":
            return LogisticRegression(class_weight="balanced", n_jobs=-1, solver="saga", max_iter=1000, **kwargs)
        
        elif model_name == "XGBoost":
            return XGBClassifier(scale_pos_weight=scale_pos_weight, n_jobs=-1, eval_metric="logloss", **kwargs)
            
        else:
            raise ValueError(f"Algorithm '{model_name}' is not supported by the ModelFactory.")

    @staticmethod
    def get_optuna_space(trial, model_name: str) -> dict:
        """Defines the Bayesian Search space for each algorithm."""
        if model_name == "RandomForest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
        
        elif model_name == "LogisticRegression":
            return {
                "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"])
            }
            
        elif model_name == "XGBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0)
            }
        else:
            raise ValueError(f"Algorithm '{model_name}' is not supported for Optuna search space definition.")