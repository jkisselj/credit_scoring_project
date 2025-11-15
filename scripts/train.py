"""
train.py

Обучение модели кредитного скоринга:
- чтение данных
- разделение X/y
- пайплайн: препроцессинг (ColumnTransformer) + модель (LightGBM/RandomForest)
- оценка AUC
- сохранение модели
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier   # запасной вариант

# LightGBM — лучше, но если не установлен, упадёт на RandomForest
try:
    from lightgbm import LGBMClassifier
    USE_LGBM = True
except ImportError:
    USE_LGBM = False

from preprocess import (
    load_application_train,
    split_features_target,
    build_preprocessor,
)


TRAIN_PATH = "data/application_train.csv"
MODEL_PATH = "results/model/my_own_model.pkl"


def main():
    print("Загружаю данные...")
    df = load_application_train(TRAIN_PATH)

    print("Делю на X и y...")
    X, y = split_features_target(df)

    print("Строю препроцессор...")
    preprocessor = build_preprocessor(X)

    print("Разделяю train/valid...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Создаю модель...")
    if USE_LGBM:
        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    print("Обучаю модель...")
    clf.fit(X_train, y_train)

    print("Считаю AUC...")
    y_valid_pred = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_valid_pred)
    print(f"AUC (validation): {auc:.4f}")

    print("Сохраняю модель...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    print(f"Готово! Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    main()
