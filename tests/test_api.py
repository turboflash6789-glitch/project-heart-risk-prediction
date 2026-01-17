# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import tempfile
import os
from pathlib import Path

# Импортируем приложение
from src.app.main import app

client = TestClient(app)


def test_health_check():
    """Тест эндпоинта /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_csv():
    sample_data = {
        "id": [1001, 1002],  # ← обязательно
        "Age": [0.5, 0.6],
        "Cholesterol": [0.4, 0.7],
        "Heart rate": [0.3, 0.2],
        "Diabetes": [0.0, 1.0],
        "Family History": [1.0, 0.0],
        "Smoking": [1.0, 0.0],
        "Obesity": [0.0, 1.0],
        "Alcohol Consumption": [1.0, 0.0],
        "Exercise Hours Per Week": [0.5, 0.2],
        "Diet": [1, 2],
        "Previous Heart Problems": [0.0, 1.0],
        "Medication Use": [0.0, 1.0],
        "Stress Level": [7.0, 9.0],
        "Sedentary Hours Per Day": [0.4, 0.6],
        "Income": [0.3, 0.5],
        "BMI": [0.4, 0.8],
        "Triglycerides": [0.5, 0.9],
        "Physical Activity Days Per Week": [3.0, 1.0],
        "Sleep Hours Per Day": [0.5, 0.3],
        "Blood sugar": [0.2, 0.3],
        "CK-MB": [0.05, 0.08],
        "Troponin": [0.04, 0.07],
        "Gender": ["Male", "Female"],
        "Systolic blood pressure": [0.4, 0.6],
        "Diastolic blood pressure": [0.5, 0.7]
    }
    df = pd.DataFrame(sample_data)
    # ... остальное без изменений
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'rb') as f:
            response = client.post("/predict", files={"file": f})
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем структуру ответа
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        
        for pred in data["predictions"]:
            assert "id" in pred
            assert "prediction" in pred
            assert pred["prediction"] in [0, 1]
            
    finally:
        os.unlink(tmp_path)


def test_predict_invalid_file():
    """Тест на загрузку не-CSSV файла."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp.write("not a csv")
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'rb') as f:
            response = client.post("/predict", files={"file": f})
        
        assert response.status_code == 400
        assert "detail" in response.json()
        
    finally:
        os.unlink(tmp_path)


def test_predict_missing_id():
    """Тест на отсутствие колонки 'id'."""
    sample_data = {
        "Age": [0.5],
        "Gender": ["Male"]
        # Нет 'id'!
    }
    
    df = pd.DataFrame(sample_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'rb') as f:
            response = client.post("/predict", files={"file": f})
        
        assert response.status_code == 400
        assert "id" in response.json()["detail"]
        
    finally:
        os.unlink(tmp_path)