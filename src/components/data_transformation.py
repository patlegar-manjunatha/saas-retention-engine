from src.helpers.initiate_stage import initiate_file
from src.helpers.exception_handling import MyException
from src.helpers.load_save import load_data, save_data, save_artifacts
from config.configurations import numerical_columns, categorical_columns, encoding_columns, target

from logging import Logger
from pandas import DataFrame
from pydantic import BaseModel

import pandas as pd
import numpy as np
import os, sys, itertools

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer



def convert_to_numeric(X):
    return X.apply(pd.to_numeric, errors="coerce")

def binary_mapper_func(df: DataFrame) -> DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].str.lower().map(
                lambda x: 1 if "yes" in str(x) else 0 if "no" in str(x) else x
            )
    return df

class TransformationConfig(BaseModel):
    data_path: str
    output_data_path: dict
    artifacts: dict


class DataTransformation:

    def __init__(self, config: TransformationConfig, logger: Logger):
        self.config = config
        self.logger = logger
    
    def binary_mapper(self, df: DataFrame) -> DataFrame:
        """
        Wrapper used only for testing compatibility.
        """
        return binary_mapper_func(df)


    def clean_data(self, df: DataFrame) -> DataFrame:
        cols = list(
            itertools.chain(
                numerical_columns,
                categorical_columns,
                encoding_columns,
                [target, "customerID"]
            )
        )
        df = df[cols].drop_duplicates().dropna().reset_index(drop=True)
        self.logger.info("Data cleaned: duplicates & null rows removed")
        return df


    def load_and_split(self) -> tuple[DataFrame, DataFrame]:
        try:
            df = load_data(self.config.data_path, logger=self.logger)
            df = self.clean_data(df)
            df[target] = df[target].map({"Yes": 1, "No": 0})

            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df[target],
            )
            save_data(
                train_df,
                test_df,
                self.config.output_data_path["preprocessed"],
                logger=self.logger,
            )
            self.logger.info("Train/Test split completed")
            return train_df, test_df
        except Exception as e:
            raise MyException(e, sys, self.logger)


    def build_preprocessor(self) -> ColumnTransformer:
        numeric_pipe = Pipeline([
            ("to_numeric", FunctionTransformer(convert_to_numeric)),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", RobustScaler()),
        ])
        binary_pipe = Pipeline([
            ("binary_mapper", FunctionTransformer(binary_mapper_func))
        ])
        categorical_pipe = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        return ColumnTransformer([
            ("num", numeric_pipe, numerical_columns),
            ("bin", binary_pipe, categorical_columns),
            ("cat", categorical_pipe, encoding_columns),
        ])


    def transform_features(self, train_df: DataFrame, test_df: DataFrame):
        X_train = train_df.drop([target, "customerID"], axis=1)
        X_test = test_df.drop([target, "customerID"], axis=1)
        y_train = train_df[target]
        y_test = test_df[target]

        preprocessor = self.build_preprocessor()
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        return np.asarray(X_train), y_train, np.asarray(X_test), y_test, preprocessor


    def save_outputs(self, X_train, y_train, X_test, y_test, preprocessor):
        transformed_dir = self.config.output_data_path["transformed"]
        os.makedirs(transformed_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(transformed_dir, "train_data.npz"),
            X=X_train,
            y=y_train,
        )
        np.savez_compressed(
            os.path.join(transformed_dir, "test_data.npz"),
            X=X_test,
            y=y_test,
        )

        save_artifacts(
            preprocessor,
            self.config.artifacts["preprocessor"],
            logger=self.logger,
        )
        self.logger.info("Preprocessor & transformed datasets saved")


    def run(self):
        train_df, test_df = self.load_and_split()
        X_train, y_train, X_test, y_test, preprocessor = self.transform_features(
            train_df,
            test_df,
        )
        self.save_outputs(X_train, y_train, X_test, y_test, preprocessor)


if __name__ == "__main__":
    params, logger = initiate_file(
        params_file_path="config/training.yaml",
        component_name="DataTransformation",
    )
    config = TransformationConfig(**params)
    DataTransformation(config, logger).run()