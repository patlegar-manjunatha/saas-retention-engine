import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.components.model_training import ModelTraining, TrainingConfig
from src.helpers.exception_handling import MyException


@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def config(tmp_path):
    return TrainingConfig(
        bucket_name="test-bucket",
        data_path="dummy_data.npz",
        selected_model="RandomForest",
        n_trials=2,
        optuna_scoring="f1",
        experiment_name="Test_Experiment",
        output_paths={
            "model": str(tmp_path / "trained_model.joblib"),
            "optuna_report": str(tmp_path / "optuna_study.json")
        }
    )

@pytest.fixture
def sample_data():
    """Generates tiny synthetic data for mocking the np.load step."""
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)
    return X, y


@patch("src.components.model_training.mlflow")
@patch("src.components.model_training.MinIOClient")
def test_initialization_success(mock_minio, mock_mlflow, config, mock_logger):
    """Ensures MinIO and MLflow are initialized correctly with config params."""
    mt = ModelTraining(config, mock_logger)
    
    mock_minio.assert_called_once()
    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once_with("Test_Experiment")

@patch("src.components.model_training.MinIOClient")
def test_initialization_minio_failure(mock_minio, config, mock_logger):
    """Ensures the pipeline crashes safely if MinIO is unreachable."""
    mock_minio.side_effect = Exception("Connection Refused")
    
    with pytest.raises(MyException):
        ModelTraining(config, mock_logger)

@patch("src.components.model_training.MinIOClient")
@patch("src.components.model_training.np.load")
def test_load_transformed_data(mock_np_load, mock_minio, config, mock_logger, sample_data):
    """Verifies that the transformed NPZ arrays are unpacked correctly."""
    mock_np_load.return_value = {'X': sample_data[0], 'y': sample_data[1]}
    
    mt = ModelTraining(config, mock_logger)
    X, y = mt.load_transformed_data()
    
    assert X.shape == (20, 5)
    assert y.shape == (20,)
    mock_np_load.assert_called_once_with("dummy_data.npz")

@patch("src.components.model_training.MinIOClient")
@patch("src.components.model_training.mlflow")
@patch("src.components.model_training.optuna")
@patch("src.components.model_training.cross_validate")
@patch("src.components.model_training.ModelFactory")
def test_optimize_hyperparameters(mock_factory, mock_cv, mock_optuna, mock_mlflow, mock_minio, config, mock_logger, sample_data):
    """Mocks Optuna and CV to ensure the objective function and nesting logic execute without training real models."""
    X, y = sample_data
    
    mock_cv.return_value = {
        'test_f1': np.array([0.9, 0.9]),
        'test_precision': np.array([0.9]),
        'test_recall': np.array([0.9]),
        'test_roc_auc': np.array([0.95])
    }
    
    mock_study = MagicMock()
    mock_study.best_value = 0.9
    mock_optuna.create_study.return_value = mock_study

    mt = ModelTraining(config, mock_logger)
    study = mt.optimize_hyperparameters(X, y)
    
    mock_optuna.create_study.assert_called_once_with(direction="maximize")
    mock_study.optimize.assert_called_once()
    assert study.best_value == 0.9

@patch("src.components.model_training.MinIOClient")
@patch("src.components.model_training.mlflow")
@patch("src.components.model_training.mlflow_sklearn") 
@patch("src.components.model_training.infer_signature")
@patch("src.components.model_training.ModelFactory")
def test_train_final_model_sklearn_branch(mock_factory, mock_infer, mock_mlflow_sklearn, mock_mlflow, mock_minio, config, mock_logger, sample_data):
    """Asserts that the Sklearn branch explicitly uses the secure 'skops' serialization."""
    X, y = sample_data
    best_params = {"max_depth": 5}
    
    mock_model = MagicMock()
    mock_factory.get_model.return_value = mock_model
    
    mt = ModelTraining(config, mock_logger)
    final_model = mt.train_final_model(X, y, best_params, 0.95)
    
    
    mock_mlflow_sklearn.log_model.assert_called_once()
    _, kwargs = mock_mlflow_sklearn.log_model.call_args
    assert kwargs["serialization_format"] == "skops"
    assert "pip_requirements" in kwargs
    mock_mlflow.log_metric.assert_called_with("cv_best_f1_score", 0.95)

@patch("src.components.model_training.MinIOClient")
@patch("src.components.model_training.mlflow")
@patch("src.components.model_training.mlflow_xgboost") 
@patch("src.components.model_training.mlflow_sklearn") 
@patch("src.components.model_training.infer_signature")
@patch("src.components.model_training.ModelFactory")
def test_train_final_model_xgboost_branch(mock_factory, mock_infer, mock_mlflow_sklearn, mock_mlflow_xgboost, mock_mlflow, mock_minio, config, mock_logger, sample_data):
    """Asserts that the pipeline routes correctly to the XGBoost native flavor."""
    config.selected_model = "XGBoost" 
    X, y = sample_data
    
    mock_model = MagicMock()
    mock_factory.get_model.return_value = mock_model
    
    mt = ModelTraining(config, mock_logger)
    mt.train_final_model(X, y, {"learning_rate": 0.1}, 0.88)

    mock_mlflow_xgboost.log_model.assert_called_once()
    mock_mlflow_sklearn.log_model.assert_not_called()


@patch("src.components.model_training.MinIOClient")
@patch("src.components.model_training.save_artifacts")
def test_save_and_upload(mock_save_artifacts, mock_minio_class, config, mock_logger):
    """Verifies that the JSON report is generated locally and both files are pushed to MinIO."""
    mock_minio_instance = MagicMock()
    mock_minio_class.return_value = mock_minio_instance
    
    mock_model = MagicMock()
    mock_study = MagicMock()
    mock_study.best_value = 0.85
    mock_study.best_params = {"n_estimators": 100}
    
    mt = ModelTraining(config, mock_logger)
    mt.save_and_upload(mock_model, mock_study)
    
    assert os.path.exists(config.output_paths["optuna_report"])
    assert mock_minio_instance.upload_file.call_count == 2

@patch("src.components.model_training.mlflow")
@patch("src.components.model_training.MinIOClient")
def test_run_pipeline_cleanup_on_exception(mock_minio, mock_mlflow, config, mock_logger):
    """
    CRITICAL MLOPS TEST: Forcibly crashes the pipeline at the data loading stage 
    to ensure the `while mlflow.active_run():` cleanup block executes.
    """
    mt = ModelTraining(config, mock_logger)
    
    mt.load_transformed_data = MagicMock(side_effect=Exception("Simulated Crash"))
    mock_mlflow.active_run.side_effect = [True, False, True, False] 
    
    with pytest.raises(MyException):
        mt.run()
        
    assert mock_mlflow.end_run.call_count == 2