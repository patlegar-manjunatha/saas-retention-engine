from src.helpers.initiate_stage import initiate_file
from src.helpers.exception_handling import MyException
from src.helpers.load_save import load_data, save_data, save_artifacts, load_yaml, log_dataset_profile
from src.storage.minio_client import MinIOClient

from logging import Logger
from pandas import DataFrame
from pydantic import BaseModel

import pandas as pd
import numpy as np
import os, sys, itertools
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

load_dotenv()

def convert_to_numeric(X: DataFrame) -> DataFrame:
    return X.apply(pd.to_numeric, errors="coerce")

def binary_mapper_func(df: DataFrame) -> DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].str.lower().map(
                lambda x: 1 if "yes" in str(x) else 0 if "no" in str(x) else x
            )
    return df

def telco_domain_logic(X: DataFrame) -> DataFrame:
    """
    Business Rule: If tenure is 0, TotalCharges must be 0.
    This is packaged inside the scikit-learn pipeline for safe inference.
    """
    X_out = X.copy()
    if "TotalCharges" in X_out.columns and "tenure" in X_out.columns:
        X_out["TotalCharges"] = pd.to_numeric(X_out["TotalCharges"], errors="coerce")
        X_out.loc[X_out["tenure"] == 0, "TotalCharges"] = 0.0
    return X_out

class TransformationConfig(BaseModel):
    data_path: str
    output_data_path: dict
    artifacts: dict
    bucket_name: str
    schema_path: str
    output_profile_path: dict


class DataTransformation:
    def __init__(self, config: TransformationConfig, logger: Logger):
        self.config = config
        self.logger = logger
        try:
            raw_schema = load_yaml(self.config.schema_path, self.logger)
            self.schema = raw_schema['data_set_schema']
            self.logger.info("Schema loaded successfully")
        except Exception as e:
            raise MyException(e, sys, self.logger)

        try:
            self.minio_client = MinIOClient(
                endpoint_url=os.getenv("END_POINT_URL", ''), 
                access_key=os.getenv('ACCESS_KEY', ''), 
                secret_key=os.getenv('SECRET_KEY', ''), 
                bucket_name=self.config.bucket_name
            )
            self.logger.debug("MinIO Client initialized for artifact uploads")
        except Exception as e:
            raise MyException(e, sys, self.logger)

    def clean_data(self, df: DataFrame) -> DataFrame:
        cols = list(
            itertools.chain(
                self.schema["numerical_columns"],
                self.schema["binary_columns"],
                self.schema["categorical_columns"],
                [self.schema["target"], self.schema["id_column"]]
            )
        )
        df = df[cols].drop_duplicates().reset_index(drop=True)
        self.logger.info("Data filtered strictly by schema definitions")
        return df

    def load_and_split(self) -> tuple[DataFrame, DataFrame]:
        try:
            df = load_data(self.config.data_path, logger=self.logger)
            df = self.clean_data(df)
            
            target_col = self.schema["target"]
            df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
            
            save_data(train_df, test_df, self.config.output_data_path["preprocessed"], logger=self.logger)
            self.logger.info("Train/Test split completed locally")
            return train_df, test_df
        except Exception as e:
            raise MyException(e, sys, self.logger)

    def build_preprocessor(self) -> Pipeline:
        numeric_pipe = Pipeline([
            ("to_numeric", FunctionTransformer(convert_to_numeric)),
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", RobustScaler()),
        ])
        binary_pipe = Pipeline([
            ("binary_mapper", FunctionTransformer(binary_mapper_func))
        ])
        categorical_pipe = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        col_transformer = ColumnTransformer([
            ("num", numeric_pipe, self.schema["numerical_columns"]),
            ("bin", binary_pipe, self.schema["binary_columns"]),
            ("cat", categorical_pipe, self.schema["categorical_columns"]),
        ])

        return Pipeline([
            ("domain_rules", FunctionTransformer(telco_domain_logic)),
            ("preprocessor", col_transformer)
        ])

    def transform_features(self, train_df: DataFrame, test_df: DataFrame):
        target_col = self.schema["target"]
        id_col = self.schema["id_column"]
        
        X_train = train_df.drop([target_col, id_col], axis=1)
        X_test = test_df.drop([target_col, id_col], axis=1)
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        master_pipeline = self.build_preprocessor()
        
        X_train_transformed = master_pipeline.fit_transform(X_train)
        X_test_transformed = master_pipeline.transform(X_test)
        
        return np.asarray(X_train_transformed), y_train, np.asarray(X_test_transformed), y_test, master_pipeline

    def save_and_upload_outputs(self, X_train, y_train, X_test, y_test, preprocessor):
        try:
            transformed_dir = self.config.output_data_path["transformed"]
            os.makedirs(transformed_dir, exist_ok=True)
            
            train_npz = os.path.join(transformed_dir, "train_data.npz")
            test_npz = os.path.join(transformed_dir, "test_data.npz")
            preprocessor_path = self.config.artifacts["preprocessor"]
            preprocessed_dir = self.config.output_data_path["preprocessed"]
            
            np.savez_compressed(train_npz, X=X_train, y=y_train)
            np.savez_compressed(test_npz, X=X_test, y=y_test)
            save_artifacts(preprocessor, preprocessor_path, logger=self.logger)

            self.logger.info("Uploading artifacts to MinIO...")
            
            self.minio_client.upload_file(os.path.join(preprocessed_dir, "train.csv"), "data/preprocessed/train.csv")
            self.minio_client.upload_file(os.path.join(preprocessed_dir, "test.csv"), "data/preprocessed/test.csv")
            
            self.minio_client.upload_file(train_npz, "data/transformed/train_data.npz")
            self.minio_client.upload_file(test_npz, "data/transformed/test_data.npz")
            
            self.minio_client.upload_file(preprocessor_path, "artifacts/preprocessing/data_preprocessor.joblib")
            self.logger.info("All artifacts successfully synced to MinIO remote")
        except Exception as e:
            raise MyException(e, sys, self.logger)

    def run(self):
        try:
            train_df, test_df = self.load_and_split()

            self.logger.info("Generating dataset profiles for train and test sets...")
            log_dataset_profile(
                df=train_df,
                dataset_name="train_data",
                output_path=self.config.output_profile_path["train"],
                bucket_name=self.config.bucket_name,
                object_key="data/preprocessed/train.csv"
            )
            log_dataset_profile(
                df=test_df,
                dataset_name="test_data",
                output_path=self.config.output_profile_path["test"],
                bucket_name=self.config.bucket_name,
                object_key="data/preprocessed/test.csv"
            )

            self.minio_client.upload_file(
                self.config.output_profile_path["train"], 
                "artifacts/reports/train_profile.json"
            )
            self.minio_client.upload_file(
                self.config.output_profile_path["test"], 
                "artifacts/reports/test_profile.json"
            )
            self.logger.info("Profiles generated and synced to MinIO.")

            X_train, y_train, X_test, y_test, preprocessor = self.transform_features(train_df, test_df)
            self.save_and_upload_outputs(X_train, y_train, X_test, y_test, preprocessor)
            self.logger.info("Data Transformation Stage Completed Successfully")
            
        except Exception as e:
            raise MyException(e, sys, self.logger)

if __name__ == "__main__":
    params, logger = initiate_file(
        params_file_path="config/training.yaml",
        component_name="DataTransformation",
    )
    config = TransformationConfig(**params)
    DataTransformation(config, logger).run()