import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from sklearn.preprocessing import StandardScaler

from src.components.data_transformation import (
    DataTransformation,
    TransformationConfig,
)


@pytest.fixture
def sample_df():
    data = {
        "customerID": [f"A{i}" for i in range(10)],
        "tenure":[1,2,3,4,5,6,7,8,9,10],
        "MonthlyCharges":[50,60,70,80,90,55,65,75,85,95],
        "TotalCharges":["50","120","210","320","450","500","650","720","810","900"],
        "SeniorCitizen":["No","Yes","No","No","Yes","No","Yes","No","No","Yes"],
        "Partner":["Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"],
        "Dependents":["No","No","Yes","No","Yes","No","Yes","No","Yes","No"],
        "PhoneService":["Yes"]*10,
        "MultipleLines":["No","Yes","No","No","Yes","No","Yes","No","No","Yes"],
        "OnlineSecurity":["No","Yes","No","Yes","No","Yes","No","Yes","No","Yes"],
        "OnlineBackup":["Yes","No","No","Yes","Yes","No","No","Yes","Yes","No"],
        "DeviceProtection":["No","Yes","No","Yes","No","Yes","No","Yes","No","Yes"],
        "TechSupport":["Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"],
        "StreamingTV":["No","Yes","No","Yes","No","Yes","No","Yes","No","Yes"],
        "StreamingMovies":["Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"],
        "PaperlessBilling":["Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"],
        "gender":["Male","Female"]*5,
        "InternetService":["DSL","Fiber optic"]*5,
        "Contract":["Month-to-month","One year"]*5,
        "PaymentMethod":["Electronic check","Mailed check"]*5,
        "Churn":["Yes","No","No","Yes","No","Yes","No","Yes","No","Yes"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def config(tmp_path):
    return TransformationConfig(
        data_path="dummy.csv",
        output_data_path={
            "preprocessed": str(tmp_path / "preprocessed"),
            "transformed": str(tmp_path / "transformed")
        },
        artifacts={
            "preprocessor": str(tmp_path / "preprocessor.pkl")
        }
    )


@pytest.fixture
def logger():
    return MagicMock()

def test_clean_data(sample_df, config, logger):
    dt = DataTransformation(config, logger)
    cleaned = dt.clean_data(sample_df)

    assert isinstance(cleaned, pd.DataFrame)
    assert cleaned.shape[0] == sample_df.shape[0]
    assert cleaned.isnull().sum().sum() == 0

def test_binary_mapper(sample_df, config, logger):
    dt = DataTransformation(config, logger)
    df = sample_df[["Partner"]]
    mapped = dt.binary_mapper(df)

    assert mapped["Partner"].isin([0,1]).all()


def test_build_preprocessor(config, logger):
    dt = DataTransformation(config, logger)
    preprocessor = dt.build_preprocessor()

    assert preprocessor is not None
    assert hasattr(preprocessor, "fit_transform")


def test_transform_features(sample_df, config, logger):
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

def test_save_outputs(config, logger):
    dt = DataTransformation(config, logger)
    X = np.random.rand(4,5)
    y = np.array([0,1,0,1])

    preprocessor = StandardScaler()
    dt.save_outputs(X, y, X, y, preprocessor)

    train_file = os.path.join(
        config.output_data_path["transformed"],
        "train_data.npz"
    )
    test_file = os.path.join(
        config.output_data_path["transformed"],
        "test_data.npz"
    )

    assert os.path.exists(train_file)
    assert os.path.exists(test_file)


def test_run_pipeline(sample_df, config, logger, monkeypatch):

    dt = DataTransformation(config, logger)

    monkeypatch.setattr(
        "src.components.data_transformation.load_data",
        lambda *args, **kwargs: sample_df
    )
    monkeypatch.setattr(
        "src.components.data_transformation.save_data",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.components.data_transformation.save_artifacts",
        lambda *args, **kwargs: None
    )

    dt.run()
    transformed_dir = config.output_data_path["transformed"]

    assert os.path.exists(os.path.join(transformed_dir, "train_data.npz"))
    assert os.path.exists(os.path.join(transformed_dir, "test_data.npz"))