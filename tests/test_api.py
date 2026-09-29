"""
Unit-тесты для REST API /predict.

Запуск: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient

from src.api import app

VALID_PAYLOAD = {
    "sex": 1,
    "age": 45,
    "hemoglobin": 150,
    "leukocytes": 6.5,
    "glucose": 5.2,
    "cholesterol": 4.8,
    "creatinine": 80,
}


@pytest.fixture(scope="module")
def client():
    """Создаёт TestClient один раз на весь модуль."""
    with TestClient(app) as c:
        yield c

def test_predict_success(client):
    """Успешный сценарий: нормальные данные → 200 + правильная структура."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["verdict"] in ("valid", "invalid")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "1.0.0"

def test_predict_invalid_sex(client):
    """sex=5 вне допустимого диапазона → 422."""
    payload = {**VALID_PAYLOAD, "sex": 5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_age_out_of_range(client):
    """age=10 ниже минимума → 422."""
    payload = {**VALID_PAYLOAD, "age": 10}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_missing_field(client):
    """Пропущено поле glucose → 422."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "glucose"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_wrong_type(client):
    """hemoglobin передан строкой → 422."""
    payload = {**VALID_PAYLOAD, "hemoglobin": "abc"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
