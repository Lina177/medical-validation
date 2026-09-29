"""
Обучение модели бинарной классификации для валидации медицинских анализов.

Использует GradientBoostingClassifier из scikit-learn с подбором порога
классификации для компенсации дисбаланса классов (95% valid / 5% invalid).

Артефакты (модель, scaler, метаданные) сохраняются в model/<version>/.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"

FEATURES = ["sex", "age", "hemoglobin", "leukocytes",
            "glucose", "cholesterol", "creatinine"]
TARGET = "is_valid"


def load_and_prepare_data(data_path: str = "data/synthetic_analyses.csv"):
    """
    Загружает датасет, разбивает на train/val/test (70/15/15),
    нормализует признаки.

    Возвращает кортеж: X_train, X_val, X_test, y_train, y_val, y_test, scaler.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)

    X = df[FEATURES].values
    y = df[TARGET].values.astype(int)

    # Разбиение: сначала 70/30, потом 30 делим пополам → 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    # Нормализация: обучаем scaler ТОЛЬКО на train, применяем ко всем
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def train_model(X_train, y_train, X_val, y_val):
    """
    Обучает GradientBoostingClassifier с учётом дисбаланса классов.

    Возвращает обученную модель.
    """
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def find_best_threshold(model, X_val, y_val):
    """
    Подбирает порог классификации по валидационной выборке,
    максимизируя F1-score. Это компенсирует дисбаланс классов.

    Возвращает оптимальный порог.
    """
    probs = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def evaluate_model(model, X, y, threshold=0.5):
    """
    Считает метрики на данных с заданным порогом.

    Метрики для класса 1 (valid) — основной класс.
    Метрики для класса 0 (invalid) — миноритарный класс (важен при дисбалансе).
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        # Класс 1 (valid) — мажоритарный
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        # Класс 0 (invalid) — миноритарный
        "precision_invalid": float(precision_score(y, preds, pos_label=0, zero_division=0)),
        "recall_invalid": float(recall_score(y, preds, pos_label=0, zero_division=0)),
        "f1_invalid": float(f1_score(y, preds, pos_label=0, zero_division=0)),
    }


def save_model(model, scaler, threshold, metrics, version=MODEL_VERSION):
    """
    Сохраняет модель, scaler и метаданные в model/<version>/.
    """
    out_dir = Path("model") / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Модель (в формате joblib)
    joblib.dump(model, out_dir / "model.joblib")

    # 2. Scaler
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # 3. Метаданные
    metadata = {
        "version": version,
        "features": FEATURES,
        "target": TARGET,
        "threshold": float(threshold),
        "metrics": metrics,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Артефакты сохранены в {out_dir}/")


def train(version: str = MODEL_VERSION):
    """
    Обучает модель и сохраняет артефакты в model/<version>/.

    Возвращает путь к сохранённой модели.
    Используется как из CLI (python src/train.py),
    так и из API (POST /retrain).
    """
    print("=" * 60)
    print("Обучение модели валидации медицинских анализов")
    print("=" * 60)

    # 1. Загрузка данных
    print("\n[1/5] Загрузка и подготовка данных...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_prepare_data()
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 2. Обучение модели
    print("\n[2/5] Обучение модели...")
    model = train_model(X_train, y_train, X_val, y_val)

    # 3. Подбор порога
    print("\n[3/5] Подбор порога по валидации...")
    best_threshold = find_best_threshold(model, X_val, y_val)
    print(f"  Оптимальный порог: {best_threshold:.3f}")

    # 4. Оценка на train/val/test
    print("\n[4/5] Оценка метрик:")
    for name, (X, y) in [("Train", (X_train, y_train)),
                         ("Val", (X_val, y_val)),
                         ("Test", (X_test, y_test))]:
        m = evaluate_model(model, X, y, threshold=best_threshold)
        print(f"  {name:5s} → acc={m['accuracy']:.4f}")
        print(f"         valid   : prec={m['precision']:.4f} | "
              f"rec={m['recall']:.4f} | f1={m['f1']:.4f}")
        print(f"         invalid : prec={m['precision_invalid']:.4f} | "
              f"rec={m['recall_invalid']:.4f} | f1={m['f1_invalid']:.4f}")

    # 5. Сохранение артефактов
    print("\n[5/5] Сохранение модели...")
    test_metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)
    save_model(model, scaler, best_threshold, test_metrics, version=version)

    print("\n" + "=" * 60)
    print(f"Готово! Модель сохранена в model/{version}/")
    print("=" * 60)

    return f"model/{version}"


if __name__ == "__main__":
    train()