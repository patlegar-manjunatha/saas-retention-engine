import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.components.data_transformation import (
    DataTransformation,
    TransformationConfig,
    telco_domain_logic,
)


@pytest.fixture
def sample_df():
    data = {
        "customerID": [f"A{i}" for i in range(10)],
        "tenure": [1, 2, 0, 4, 5, 0, 7, 8, 9, 10],  # Added 0s to test domain logic
        "MonthlyCharges": [50, 60, 70, 80, 90, 55, 65, 75, 85, 95],
        "TotalCharges": ["50", "120", " ", "320", "450", " ", "650", "720", "810", "900"], # Spaces simulate missing total charges for 0 tenure
        "SeniorCitizen": ["No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
        "Partner": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "PhoneService": ["Yes"] * 10,
        "MultipleLines": ["No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
        "OnlineSecurity": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        "TechSupport": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "StreamingTV": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        "StreamingMovies": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "gender": ["Male", "Female"] * 5,
        "InternetService": ["DSL", "Fiber optic"] * 5,
        "Contract": ["Month-to-month", "One year"] * 5,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 5,
        "Churn": ["Yes", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_schema():
    return {
        "id_column": "customerID",
        "target": "Churn",
        "numerical_columns": ["TotalCharges", "MonthlyCharges", "tenure"],
        "binary_columns": [
            "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "PaperlessBilling"
        ],
        "categorical_columns": ["gender", "InternetService", "Contract", "PaymentMethod"]
    }

@pytest.fixture
def config(tmp_path):
    return TransformationConfig(
        data_path="dummy.csv",
        output_data_path={
            "preprocessed": str(tmp_path / "preprocessed"),
            "transformed": str(tmp_path / "transformed")
        },
        output_profile_path={                                   
            "train": str(tmp_path / "train_profile.json"),
            "test": str(tmp_path / "test_profile.json")
        },
        artifacts={
            "preprocessor": str(tmp_path / "preprocessor.joblib")
        },
        bucket_name="test-bucket",
        schema_path="dummy_schema.yaml"
    )
@pytest.fixture
def logger():
    return MagicMock()


def test_telco_domain_logic():
    """Tests the isolated business rule to ensure missing TotalCharges are set to 0.0 when tenure is 0"""
    df = pd.DataFrame({
        "tenure": [10, 0, 5],
        "TotalCharges": ["100.5", " ", "50.0"]
    })
    transformed_df = telco_domain_logic(df)
    
    assert transformed_df.loc[1, "TotalCharges"] == 0.0
    assert transformed_df.loc[0, "TotalCharges"] == 100.5


@patch("src.components.data_transformation.MinIOClient")
@patch("src.components.data_transformation.load_yaml")
def test_clean_data(mock_load_yaml, mock_minio, sample_df, mock_schema, config, logger):
    mock_load_yaml.return_value = {"data_set_schema": mock_schema}
    
    dt = DataTransformation(config, logger)
    cleaned = dt.clean_data(sample_df)

    assert isinstance(cleaned, pd.DataFrame)
    assert cleaned.shape[0] == sample_df.shape[0]
    assert "customerID" in cleaned.columns
    assert "Churn" in cleaned.columns


@patch("src.components.data_transformation.MinIOClient")
@patch("src.components.data_transformation.load_yaml")
def test_build_preprocessor(mock_load_yaml, mock_minio, mock_schema, config, logger):
    mock_load_yaml.return_value = {"data_set_schema": mock_schema}
    
    dt = DataTransformation(config, logger)
    master_pipeline = dt.build_preprocessor()

    assert master_pipeline is not None
    assert isinstance(master_pipeline, Pipeline)
    assert "domain_rules" in master_pipeline.named_steps
    assert "preprocessor" in master_pipeline.named_steps


@patch("src.components.data_transformation.MinIOClient")
@patch("src.components.data_transformation.load_yaml")
def test_transform_features(mock_load_yaml, mock_minio, sample_df, mock_schema, config, logger):
    mock_load_yaml.return_value = {"data_set_schema": mock_schema}
    
    dt = DataTransformation(config, logger)
    train_df = sample_df.copy()
    test_df = sample_df.copy()
    
    X_train, y_train, X_test, y_test, preprocessor = dt.transform_features(
        train_df, test_df
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert len(y_train) == len(train_df)
    assert preprocessor is not None


@patch("src.components.data_transformation.save_artifacts")
@patch("src.components.data_transformation.MinIOClient")
@patch("src.components.data_transformation.load_yaml")
def test_save_and_upload_outputs(mock_load_yaml, mock_minio_class, mock_save_artifacts, mock_schema, config, logger):
    mock_load_yaml.return_value = {"data_set_schema": mock_schema}
    
    mock_minio_instance = MagicMock()
    mock_minio_class.return_value = mock_minio_instance
    
    dt = DataTransformation(config, logger)
    X = np.random.rand(4, 5)
    y = np.array([0, 1, 0, 1])

    preprocessor = StandardScaler()
    dt.save_and_upload_outputs(X, y, X, y, preprocessor)

    train_file = os.path.join(config.output_data_path["transformed"], "train_data.npz")
    test_file = os.path.join(config.output_data_path["transformed"], "test_data.npz")
    assert os.path.exists(train_file)
    assert os.path.exists(test_file)
    mock_save_artifacts.assert_called_once()

    assert mock_minio_instance.upload_file.call_count == 5


@patch("src.components.data_transformation.log_dataset_profile") 
@patch("src.components.data_transformation.save_data")
@patch("src.components.data_transformation.load_data")
@patch("src.components.data_transformation.MinIOClient")
@patch("src.components.data_transformation.load_yaml")
def test_run_pipeline(mock_load_yaml, mock_minio_class, mock_load_data, mock_save_data, mock_log_profile, sample_df, mock_schema, config, logger, monkeypatch):
    mock_load_yaml.return_value = {"data_set_schema": mock_schema}
    mock_load_data.return_value = sample_df
    
    mock_minio_instance = MagicMock()
    mock_minio_class.return_value = mock_minio_instance

    monkeypatch.setattr("src.components.data_transformation.save_artifacts", lambda *args, **kwargs: None)

    dt = DataTransformation(config, logger)
    dt.run()

    transformed_dir = config.output_data_path["transformed"]

    assert os.path.exists(os.path.join(transformed_dir, "train_data.npz"))
    assert os.path.exists(os.path.join(transformed_dir, "test_data.npz"))
    assert mock_log_profile.call_count == 2