# src/app/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import tempfile
import os
from pathlib import Path

# Импорты из твоей библиотеки
from src.core.data_processor import DataProcessor
from src.core.model_wrapper import ModelWrapper

# Инициализация FastAPI
app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="Предсказывает риск сердечного приступа на основе медицинских данных пациента.",
    version="1.0.0"
)

# Пути к артефактам (относительно корня проекта)
BASE_DIR = Path(__file__).parent.parent.parent  # project-heart-risk-prediction/
MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# Загрузка модели и препроцессора при старте
try:
    processor = DataProcessor(encoder_path=str(ENCODER_PATH))
    model = ModelWrapper(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель или encoder: {e}")


@app.post("/predict", summary="Предсказание риска")
async def predict(file: UploadFile = File(...)):
    """
    Принимает CSV-файл с данными пациентов и возвращает предсказания.
    
    Ожидаемые колонки в CSV:
    - Все признаки из исходного датасета (включая 'id' и 'Gender')
    - Колонка 'id' обязательна для сопоставления
    
    Возвращает JSON с массивом объектов: {"id": ..., "prediction": 0/1}
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате .csv")

    try:
        # Сохраняем загруженный файл во временный CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Читаем данные
        df = pd.read_csv(tmp_path)

        # Проверяем наличие 'id'
        if 'id' not in df.columns:
            raise HTTPException(status_code=400, detail="В CSV должна быть колонка 'id'")

        # Сохраняем id для ответа
        ids = df['id'].copy()

        # Предобработка
        df_processed = processor.transform(df)

        # Предсказание
        predictions = model.predict(df_processed)

        # Формируем результат
        result = [
            {"id": int(id_val), "prediction": int(pred)}
            for id_val, pred in zip(ids, predictions)
        ]

        return JSONResponse(content={"predictions": result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    finally:
        # Удаляем временный файл
        if 'tmp_path' in locals():
            os.unlink(tmp_path)


@app.get("/health", summary="Проверка работоспособности")
async def health_check():
    """Возвращает статус 'ok', если сервис запущен."""
    return {"status": "ok"}