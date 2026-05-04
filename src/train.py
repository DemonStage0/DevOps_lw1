"""Модуль обучения модели классификации типов стекла."""

import configparser
import hashlib
import os
import pickle
import sys
import traceback
from datetime import datetime
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocess import DataPreprocessor
from logger import Logger

SHOW_LOG = True


class ModelTrainer:
    """Класс для обучения модели и управления экспериментами."""

    def __init__(self) -> None:
        """Инициализация тренера модели с конфигурацией."""
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.exp_path = self.config["PATHS"]["experiments"]
        os.makedirs(self.exp_path, exist_ok=True)
        self.log.info("ModelTrainer инициализирован")

    def train(self) -> dict:
        """
        Обучение модели RandomForestClassifier с системой экспериментов.

        Returns:
            dict: результат обучения с ключами status, f1_score, experiment_path.
        """
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        params = {
            'n_estimators': self.config.getint("MODEL", "n_estimators"),
            'criterion': self.config["MODEL"]["criterion"],
            'max_depth': self.config.getint("MODEL", "max_depth"),
            'random_state': self.config.getint("MODEL", "random_state")
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = {
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'accuracy': accuracy_score(y_test, y_pred)
        }

        self.log.info(f"Модель обучена. Метрики: {metrics}")

        exp_dir = self._save_experiment(model, scaler, params, metrics)
        self.log.info(f"Эксперимент сохранён: {exp_dir}")

        return {
            'status': 'success',
            'f1_score': metrics['f1_score'],
            'experiment_path': exp_dir
        }

    def _save_experiment(self, model, scaler, params: dict,
                         metrics: dict) -> str:
        """
        Сохранение эксперимента с моделью, метриками и конфигурацией.

        Args:
            model: обученная модель.
            scaler: масштабировщик данных.
            params (dict): параметры модели.
            metrics (dict): метрики модели.

        Returns:
            str: путь к директории эксперимента.
        """
        exp_num = len(os.listdir(self.exp_path)) + 1
        exp_dir = os.path.join(self.exp_path, f"exp_{exp_num}")
        os.makedirs(exp_dir, exist_ok=True)

        model_path = os.path.join(exp_dir, "trained_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)

        with open(model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()

        config_data = {
            'model_type': self.config["MODEL"]["type"],
            'parameters': params,
            'data_path': self.config["PATHS"]["data"],
            'model_hash': model_hash,
            'log_dir': exp_dir,
            'timestamp': datetime.now().isoformat()
        }

        metrics_data = {
            'metrics': metrics,
            'model_path': model_path,
            'data_path': self.config["PATHS"]["data"],
            'timestamp': datetime.now().isoformat()
        }

        for filename, data in [
            ("config.yml", config_data),
            ("metrics.yml", metrics_data)
        ]:
            with open(os.path.join(exp_dir, filename), 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

        log_path = os.path.join(exp_dir, "logs.txt")
        with open("logfile.log", 'r') as src, open(log_path, 'w') as dst:
            dst.write(src.read())

        return exp_dir