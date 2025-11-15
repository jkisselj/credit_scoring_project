# Credit scoring project

Проект по построению модели кредитного скоринга: предсказание вероятности дефолта по данным Home Credit.

## Структура

- data/ — сырые данные (CSV из архива home-credit-default-risk).
- results/
  - model/ — обученная модель и отчет (my_own_model.pkl, model_report.txt).
  - feature_engineering/ — ноутбук EDA.ipynb.
  - clients_outputs/ — визуализации по выбранным клиентам.
  - dashboard/ — код дашборда (опционально).
- scripts/ — python-скрипты (preprocess.py, train.py, predict.py).

## Как запустить

1. Создать и активировать виртуальное окружение:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Запустить EDA:
   - открыть `results/feature_engineering/EDA.ipynb` в Jupyter Notebook или Jupyter Lab.

## Kaggle username

Смотри файл username.txt.
