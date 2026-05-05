import os
import sys
import unittest
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from preprocess import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Тесты для DataPreprocessor."""

    def setUp(self) -> None:
        """Подготовка перед каждым тестом."""
        self.preprocessor = DataPreprocessor()

    def test_prepare_data(self):
        """Проверка корректности разделения данных."""
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data()
        self.assertEqual(X_train.shape[1], 9)
        self.assertEqual(X_test.shape[1], 9)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertGreater(len(X_train), len(X_test))


if __name__ == "__main__":
    unittest.main()