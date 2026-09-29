"""
Генерация синтетического датасета
лабораторных анализов.

Имитирует реальные медицинские данные:
- показатели зависят от пола и
возраста;
- ~95% анализов корректны (is_valid=1),
~5% - нет (is_valid=0);
- у "некорректных" анализов показатели выходят за пределы нормы.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_SAMPLES = 12_000
INVALID_FRACTION = 0.05

# Референсные интервалы для "здорового" пациента:
# (среднее, стандартное отклонение,минимум, максимум)
REFERENCE = {
 "hemoglobin":  (145.0, 15.0, 100.0, 180.0),
    "leukocytes":  (6.5,   1.8,   3.5,  10.0),
    "glucose":     (5.0,   0.6,   3.9,   6.1),
    "cholesterol": (4.8,   0.9,   3.0,   6.2),
    "creatinine":  (85.0,  15.0,  60.0, 110.0),
}
# Поправки на пол: у мужчин гемоглобин и креатинин выше
SEX_SHIFT = {
    "hemoglobin":  {0: -12.0, 1: +12.0},
    "creatinine":  {0: -12.0, 1: +12.0},
    "leukocytes":  {0: 0.0,   1: 0.0},
    "glucose":     {0: 0.0,   1: 0.0},
    "cholesterol": {0: 0.0,   1: 0.0},
}

# Поправки на возраст: на каждые 10 лет от 40 лет
AGE_TREND = {
    "hemoglobin":  -1.0,
    "leukocytes":   0.0,
    "glucose":     +0.15,
    "cholesterol": +0.20,
    "creatinine":  +0.5,
}


def _clip(a, lo, hi):
    """Обрезает значения массива a до диапазона [lo, hi]."""
    return np.clip(a, lo, hi)


def generate_dataset(n_samples: int = N_SAMPLES,
                     invalid_fraction: float = INVALID_FRACTION,
                     seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Генерирует синтетический датасет лабораторных анализов.

    Возвращает DataFrame с колонками:
    sex, age, hemoglobin, leukocytes, glucose, cholesterol, creatinine, is_valid.
    """
    rng = np.random.default_rng(seed)

    # --- Пол и возраст ---
    sex = rng.integers(0, 2, size=n_samples)
    age = rng.integers(18, 91, size=n_samples)

    # --- Кто "некорректный" ---
    n_invalid = int(round(n_samples * invalid_fraction))
    is_valid = np.ones(n_samples, dtype=np.int8)
    invalid_idx = rng.choice(n_samples, size=n_invalid, replace=False)
    is_valid[invalid_idx] = 0

    # --- Генерация показателей ---
    features = {}
    for feat, (mean, std, lo, hi) in REFERENCE.items():
        base = rng.normal(mean, std, size=n_samples)

        sex_shift = np.array([SEX_SHIFT[feat][s] for s in sex])
        base += sex_shift

        base += AGE_TREND[feat] * (age - 40) / 10.0

        shift = np.zeros(n_samples)
        direction = rng.choice([-1.0, 1.0], size=n_invalid)
        magnitude = rng.uniform(2.0, 4.0, size=n_invalid) * std
        shift[invalid_idx] = direction * magnitude
        base += shift

        features[feat] = _clip(base, lo * 0.5, hi * 1.5)

    df = pd.DataFrame({
        "sex": sex,
        "age": age,
        "hemoglobin": np.round(features["hemoglobin"], 1),
        "leukocytes": np.round(features["leukocytes"], 2),
        "glucose": np.round(features["glucose"], 2),
        "cholesterol": np.round(features["cholesterol"], 2),
        "creatinine": np.round(features["creatinine"], 1),
        "is_valid": is_valid,
    })
    return df


def train_val_test_split(df: pd.DataFrame,
                         seed: int = RANDOM_STATE):
    """Разбивает DataFrame на train/val/test (70/15/15) со стратификацией."""
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["is_valid"], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["is_valid"], random_state=seed
    )
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    df.to_csv(out_dir / "synthetic_analyses.csv", index=False)

    train, val, test = train_val_test_split(df)

    print(f"Всего записей: {len(df)}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("\nРаспределение классов (is_valid):")
    print(df["is_valid"].value_counts(normalize=True).round(4))
    print("\nПервые 5 строк:")
    print(df.head())