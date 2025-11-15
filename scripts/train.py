"""
train.py

Обучение модели кредитного скоринга.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib

from preprocess import load_application_train, basic_preprocess

def main():
    # TODO: заменить путь на путь внутри data/
    train_path = "data/application_train.csv"
    df = load_application_train(train_path)

    # TODO: доработать препроцессинг и выбор фичей/target
    df = basic_preprocess(df)

    # Заглушка для примера: здесь нужно выбрать X, y
    # X = ...
    # y = df["TARGET"]

    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)

    # y_pred_proba = model.predict_proba(X_valid)[:, 1]
    # auc = roc_auc_score(y_valid, y_pred_proba)
    # print(f"AUC: {auc:.4f}")

    # joblib.dump(model, "results/model/my_own_model.pkl")

if __name__ == "__main__":
    main()
