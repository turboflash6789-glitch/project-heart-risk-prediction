# src/core/model_wrapper.py

import pandas as pd
import joblib
from pathlib import Path


class ModelWrapper:
    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction").astype(int)