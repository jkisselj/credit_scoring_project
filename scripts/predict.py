"""
predict.py

Скрипт для генерации предсказаний на тестовом наборе для Kaggle.
"""

import pandas as pd
import joblib
from preprocess import load_application_test, basic_preprocess

def main():
    # TODO: заменить путь на путь внутри data/
    test_path = "data/application_test.csv"
    df_test = load_application_test(test_path)
    df_test_processed = basic_preprocess(df_test)

    # TODO: подставить правильные имена признаков и ID клиента
    # model = joblib.load("results/model/my_own_model.pkl")
    # y_test_proba = model.predict_proba(df_test_processed[feature_cols])[:, 1]

    # submission = pd.DataFrame({
    #     "SK_ID_CURR": df_test["SK_ID_CURR"],
    #     "TARGET": y_test_proba
    # })
    # submission.to_csv("results/model/kaggle_submission.csv", index=False)

if __name__ == "__main__":
    main()
