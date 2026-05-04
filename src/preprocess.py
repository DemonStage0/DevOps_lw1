import configparser
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import Logger

SHOW_LOG = True


class DataPreprocessor:
    """Класс для загрузки, разделения и сохранения данных."""

    def __init__(self) -> None:
        """Инициализация препроцессора с конфигурацией из config.ini."""
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.data_path = self.config["PATHS"]["data"]
        self.exp_path = self.config["PATHS"]["experiments"]
        self.log.info("DataPreprocessor инициализирован")

    def prepare_data(self) -> tuple:
        """
        Загрузка и разделение датасета на train/test.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        dataset = pd.read_csv(self.data_path)
        self.log.info(f"Датасет загружен: {dataset.shape}")

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        test_size = self.config.getfloat("MODEL", "test_size")
        random_state = self.config.getint("MODEL", "random_state")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=random_state, stratify=y
        )

        self.log.info(
            f"Данные разделены: train={X_train.shape}, test={X_test.shape}"
        )
        return X_train, X_test, y_train, y_test