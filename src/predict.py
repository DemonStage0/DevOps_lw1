import os
import pickle
import numpy as np
import pandas as pd

from logger import Logger

SHOW_LOG = True


class GlassPredictor:
    """Класс для загрузки модели и выполнения предсказаний."""

    def __init__(self) -> None:
        """Инициализация предиктора."""
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.model = None
        self.scaler = None
        self.log.info("GlassPredictor инициализирован")

    def load_latest_model(self) -> bool:
        """
        Загрузка последней сохранённой модели из директории experiments.

        Returns:
            bool: True если модель загружена, иначе False.
        """
        exp_path = "experiments"
        if not os.path.exists(exp_path):
            self.log.error("Директория experiments не найдена")
            return False

        experiments = sorted(
            [d for d in os.listdir(exp_path)
             if d.startswith("exp_")],
            key=lambda x: int(x.split("_")[1])
        )

        if not experiments:
            self.log.error("Нет сохранённых экспериментов")
            return False

        latest_exp = experiments[-1]
        model_path = os.path.join(exp_path, latest_exp, "trained_model.pkl")

        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.log.info(f"Модель загружена: {model_path}")
            return True
        except FileNotFoundError:
            self.log.error(f"Файл модели не найден: {model_path}")
            return False

    def predict(self, features: list) -> int:
        """
        Предсказание класса стекла по заданным признакам.

        Args:
            features (list): 9 признаков стекла.

        Returns:
            int: предсказанный класс (1-7) или None при ошибке.
        """
        if self.model is None and not self.load_latest_model():
            return None

        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(pd.DataFrame(X, columns=self.scaler.feature_names_in_))
        prediction = int(self.model.predict(X_scaled)[0])

        self.log.info(f"Предсказание: {prediction} для {features}")
        return prediction