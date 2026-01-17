# src/core/data_processor.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


class DataProcessor:
    """
    Класс для предобработки данных перед предсказанием.
    
    Поддерживает:
    - Удаление служебных колонок ('id', 'Unnamed: 0')
    - Унификацию значений пола (Gender) к 'Male'/'Female'
    - Кодирование Gender с помощью сохранённого или нового LabelEncoder
    """

    def __init__(self, encoder_path: str = None):
        """
        Инициализация процессора.
        
        Parameters:
            encoder_path (str, optional): Путь к сохранённому файлу LabelEncoder (.pkl).
                                          Если указан и файл существует — загружается.
                                          Если None — создаётся новый encoder (для обучения).
        """
        if encoder_path and Path(encoder_path).exists():
            self.label_encoder = joblib.load(encoder_path)
            self._is_fitted = True
        else:
            self.label_encoder = LabelEncoder()
            self._is_fitted = False

    @staticmethod
    def _unify_gender(value) -> str:
        """
        Приводит разные форматы значения пола к единообразному виду.
        
        Поддерживаемые входы:
          - 'Male', 'Female'
          - '1.0', '0.0', '1', '0'
          
        Возвращает: 'Male' или 'Female'
        """
        str_val = str(value).strip()
        if str_val in ['1.0', '1', 'Male']:
            return 'Male'
        elif str_val in ['0.0', '0', 'Female']:
            return 'Female'
        else:
            # На случай неожиданных значений — оставляем как есть,
            # но это может вызвать ошибку при кодировании
            return str_val

    def fit(self, df: pd.DataFrame):
        """
        Обучает внутренний LabelEncoder на колонке 'Gender'.
        
        Parameters:
            df (pd.DataFrame): DataFrame с колонкой 'Gender'
            
        Returns:
            self
        """
        df = df.copy()
        df['Gender'] = df['Gender'].apply(self._unify_gender)
        self.label_encoder.fit(df['Gender'])
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет предобработку к данным.
        
        Выполняет:
          - Удаление колонок 'id' и 'Unnamed: 0' (если есть)
          - Унификацию и кодирование 'Gender'
          
        Parameters:
            df (pd.DataFrame): Исходный DataFrame
            
        Returns:
            pd.DataFrame: Обработанный DataFrame без служебных колонок и с числовым 'Gender'
            
        Raises:
            ValueError: Если encoder не обучен (fit не вызван)
        """
        if not self._is_fitted:
            raise ValueError("DataProcessor не обучен. Вызовите .fit() или укажите encoder_path при инициализации.")

        df = df.copy()

        # Удаляем служебные колонки, если они присутствуют
        cols_to_drop = ['id', 'Unnamed: 0']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Унифицируем и кодируем Gender
        df['Gender'] = df['Gender'].apply(self._unify_gender)
        df['Gender'] = self.label_encoder.transform(df['Gender'])

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Выполняет fit и transform за один вызов."""
        return self.fit(df).transform(df)