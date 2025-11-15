"""
preprocess.py

Функции для загрузки данных и подготовки фичей.
"""

import pandas as pd

def load_application_train(path: str) -> pd.DataFrame:
    \"\"\"Загрузка application_train.csv\"\"\"
    return pd.read_csv(path)

def load_application_test(path: str) -> pd.DataFrame:
    \"\"\"Загрузка application_test.csv\"\"\"
    return pd.read_csv(path)

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Простейший препроцессинг (заглушка, доработать позже).\"\"\"
    # TODO: реализовать фичи и обработку пропусков
    return df
