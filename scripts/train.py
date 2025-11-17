
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from scripts.preprocess import (
    load_application_train,
    split_features_target,
    build_preprocessor,
)


def main():
    train_path = "data/application_train.csv"
    df = load_application_train(train_path)

    # Разделяем на X и y
    X, y = split_features_target(df)

    # Строим препроцессор на основе треновых данных
    preprocessor = build_preprocessor(X)

    # Модель (пока простой RandomForest — потом можно заменить на LightGBM/XGBoost)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
    )

    # Общий sklearn-пайплайн: препроцессор + модель
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Трейн/валидация
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Обучаем
    clf.fit(X_train, y_train)

    # Оцениваем AUC на валидации
    y_valid_proba = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_valid_proba)
    print(f"AUC на валидации: {auc:.4f}")

    # Сохраняем всю пайплайн-модель (и препроцесс, и сам RandomForest)
    joblib.dump(clf, "results/model/my_own_model.pkl")
    print("Модель сохранена в results/model/my_own_model.pkl")


if __name__ == "__main__":
    main()
