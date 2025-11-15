"""
preprocess.py

Функции для загрузки данных и подготовки фичей.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"


def load_application_train(path: str) -> pd.DataFrame:
    """Загрузка application_train.csv"""
    return pd.read_csv(path)


def load_application_test(path: str) -> pd.DataFrame:
    """Загрузка application_test.csv"""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    """
    Разделяем на X и y.
    ID и TARGET не включаем в X.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"В датафрейме нет колонки {TARGET_COL}")

    y = df[TARGET_COL]
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    X = df[feature_cols]
    return X, y


def get_feature_lists(X: pd.DataFrame):
    """
    Возвращаем списки числовых и категориальных колонок.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Создаёт препроцессор:
    - числовые: медиана
    - категориальные: мода + one-hot encoding
    """
    numeric_cols, categorical_cols = get_feature_lists(X)

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor
