
import os
import pandas as pd
import joblib

from scripts.preprocess import load_application_test, ID_COL

def main():
    # Корень проекта
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Путь к тестовым данным
    test_path = os.path.join(project_root, "data", "application_test.csv")
    
    # Загружаем test
    df_test = load_application_test(test_path)

    # Пока фичи — все колонки кроме ID
    feature_cols = [c for c in df_test.columns if c != ID_COL]
    X_test = df_test[feature_cols]

    # Путь к модели
    model_path = os.path.join(project_root, "results", "model", "my_own_model.pkl")
    model = joblib.load(model_path)

    # Предсказания вероятности дефолта
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # DataFrame для Kaggle
    submission = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        "TARGET": y_test_proba,
    })

    # Куда сохранить
    out_path = os.path.join(project_root, "results", "model", "kaggle_submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")


if __name__ == "__main__":
    main()
