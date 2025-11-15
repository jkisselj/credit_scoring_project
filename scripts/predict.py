"""
predict.py

Генерация CSV для Kaggle submission.
"""

import os
import joblib
import pandas as pd

from preprocess import (
    load_application_test,
    ID_COL,
    split_features_target,
)


TEST_PATH = "data/application_test.csv"
MODEL_PATH = "results/model/my_own_model.pkl"
SUBMISSION_PATH = "results/model/kaggle_submission.csv"


def main():
    print("Загружаю test данные...")
    df_test = load_application_test(TEST_PATH)

    print("Загружаю модель...")
    model = joblib.load(MODEL_PATH)

    print("Готовлю признаки...")
    feature_cols = [c for c in df_test.columns if c != ID_COL]
    X_test = df_test[feature_cols]

    print("Считаю вероятности...")
    y_pred = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        "TARGET": y_pred,
    })

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Сабмишн сохранён: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
