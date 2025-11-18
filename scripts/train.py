
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from scripts.preprocess import (
    load_application_train,
    split_features_target,
    build_preprocessor,
)


def main():
    # Корень проекта = папка на уровень выше scripts/
    project_root = os.path.dirname(os.path.dirname(__file__))

    # Путь к train-данным
    train_path = os.path.join(project_root, "data", "application_train.csv")

    # Загружаем данные
    df = load_application_train(train_path)

    # Разделяем на X и y
    X, y = split_features_target(df)

    # Train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Препроцессор по тренировочным данным
    preprocessor = build_preprocessor(X_train)

    # Модель (RandomForest, как в ноутбуке)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
    )

    # Пайплайн: препроцессинг → модель
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf_model),
        ]
    )

    # Обучение
    clf.fit(X_train, y_train)

    # Предсказание на валидации
    y_valid_proba = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_valid_proba)
    print(f"AUC на валидации (train.py): {auc:.4f}")

    # Сохраняем весь пайплайн (preprocessor + model)
    model_path = os.path.join(project_root, "results", "model", "my_own_model.pkl")
    joblib.dump(clf, model_path)
    print(f"Модель сохранена в {model_path}")


if __name__ == "__main__":
    main()
