"""
REST API для инференса модели валидации медицинских анализов.

Эндпоинт:
- POST /predict — принимает признаки анализа, возвращает verdict + confidence.

Запуск:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------- Логирование ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medical_validation.api")


class ModelBundle:
    """Хранит модель, scaler и метаданные в памяти."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.metadata: dict = {}
        self.version: str = ""

    def load(self, version: str = "1.0.0") -> None:
        """Загружает модель, scaler и метаданные из model/<version>/."""
        model_dir = Path("model") / version

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Модель версии {version} не найдена в {model_dir}"
            )

        # Загружаем модель и scaler
        self.model = joblib.load(model_dir / "model.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")

        # Загружаем метаданные
        with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.version = version
        logger.info("Модель версии %s загружена из %s", version, model_dir)

    @property
    def threshold(self) -> float:
        """Порог классификации из метаданных."""
        return float(self.metadata.get("threshold", 0.5))

    @property
    def features(self) -> list:
        """Список признаков из метаданных."""
        return list(self.metadata.get("features", []))


# Глобальный экземпляр
bundle = ModelBundle()


# ---------- Pydantic-схемы ----------

class PredictRequest(BaseModel):
    """Схема входных данных для эндпоинта /predict."""
    sex: int = Field(..., ge=0, le=1, description="0 – женский, 1 – мужской")
    age: int = Field(..., ge=18, le=90, description="Возраст, лет")
    hemoglobin: float = Field(..., ge=30, le=250, description="Гемоглобин, г/л")
    leukocytes: float = Field(..., ge=0.1, le=50, description="Лейкоциты, 10^9/л")
    glucose: float = Field(..., ge=1.0, le=40, description="Глюкоза, ммоль/л")
    cholesterol: float = Field(..., ge=1.0, le=20, description="Холестерин, ммоль/л")
    creatinine: float = Field(..., ge=10, le=1500, description="Креатинин, мкмоль/л")


class PredictResponse(BaseModel):
    """Схема ответа эндпоинта /predict."""
    verdict: str = Field(..., description='"valid" или "invalid"')
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Вероятность класса valid")
    model_version: str = Field(..., description="Версия модели")

# ---------- FastAPI приложение ----------

app = FastAPI(
    title="Medical Analysis Validation API",
    description="Сервис валидации медицинских анализов (ML-модель)",
    version="1.0.0",
)

@app.on_event("startup")
def startup_event():
    """Загружает модель при старте приложения (один раз)."""
    logger.info("Запуск приложения, загрузка модели...")
    bundle.load(version="1.0.0")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логирует каждый HTTP-запрос и его статус."""
    logger.info("→ %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("Ошибка при обработке запроса: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    logger.info("← %s %s %s",
                request.method, request.url.path, response.status_code)
    return response

# ---------- Эндпоинт /predict ----------

def _predict_proba(payload: PredictRequest) -> float:
    """Прогоняет признаки через модель, возвращает вероятность класса valid."""
    if bundle.model is None or bundle.scaler is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    # Формируем вектор признаков в порядке FEATURES
    features = np.array([[
        payload.sex,
        payload.age,
        payload.hemoglobin,
        payload.leukocytes,
        payload.glucose,
        payload.cholesterol,
        payload.creatinine,
    ]], dtype=float)

    # Нормализация тем же scaler, что использовался при обучении
    features_scaled = bundle.scaler.transform(features)

    # Вероятность класса 1 (valid)
    prob = float(bundle.model.predict_proba(features_scaled)[0, 1])
    return prob

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """
    Принимает признаки одного анализа, возвращает verdict + confidence.
    """
    try:
        prob = _predict_proba(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ошибка инференса: %s", exc)
        raise HTTPException(status_code=500, detail="Ошибка инференса") from exc

    verdict = "valid" if prob >= bundle.threshold else "invalid"
    return PredictResponse(
        verdict=verdict,
        confidence=round(prob, 4),
        model_version=bundle.version,
    )


# ---------- Корневой эндпоинт ----------

@app.get("/")
def root():
    """Приветственная страница API."""
    return {
        "message": "Medical Analysis Validation API",
        "description": "Сервис валидации медицинских анализов (ML-модель)",
        "version": bundle.version or "1.0.0",
        "docs": "/docs",
        "predict": "/predict",
    }




# ---------- Эндпоинт /retrain ----------

class RetrainRequest(BaseModel):
    """Схема запроса на переобучение модели."""
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$",
                         description="Новая версия, например '1.1.0'")


class RetrainResponse(BaseModel):
    """Схема ответа /retrain."""
    status: str
    new_version: str
    message: str


@app.post("/retrain", response_model=RetrainResponse, status_code=202)
def retrain(payload: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Запускает переобучение модели в фоне и сохраняет новую версию.

    Возвращает 202 Accepted сразу — обучение идёт асинхронно.
    Проверь логи сервера, чтобы увидеть завершение.
    """
    from src.train import train as run_training

    logger.info("Запуск переобучения модели → версия %s", payload.version)

    def _do_retrain(version: str):
        try:
            path = run_training(version=version)
            logger.info("Переобучение завершено: %s", path)
            bundle.load(version=version)
            logger.info("Активная модель переключена на версию %s", version)
        except Exception as exc:
            logger.exception("Ошибка переобучения: %s", exc)

    background_tasks.add_task(_do_retrain, payload.version)

    return RetrainResponse(
        status="accepted",
        new_version=payload.version,
        message="Обучение запущено в фоне. Проверьте логи сервера через 30-60 секунд.",
    )