"""API тесты для FastAPI микросервиса."""

import os
import sys
import shutil
import unittest
from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from app import app
from predict import GlassPredictor


class TestAPI(unittest.TestCase):
    """Тесты для API эндпоинтов."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        """Очистка модели и experiments для изоляции тестов."""
        from app import predictor
        predictor.model = None
        predictor.scaler = None

        exp_path = "experiments"
        if os.path.exists(exp_path):
            for item in os.listdir(exp_path):
                item_path = os.path.join(exp_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_predict_no_model(self):
        response = self.client.get(
            "/predict",
            params={
                "RI": 1.52101, "Na": 13.64, "Mg": 4.49, "Al": 1.1,
                "Si": 71.78, "K": 0.06, "Ca": 8.75, "Ba": 0.0, "Fe": 0.0
            }
        )
        self.assertEqual(response.status_code, 404)

    def test_train(self):
        response = self.client.get("/train")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Модель обучена успешно", response.json()["message"])

    def test_predict_after_train(self):
        self.client.get("/train")
        response = self.client.get(
            "/predict",
            params={
                "RI": 1.52101, "Na": 13.64, "Mg": 4.49, "Al": 1.1,
                "Si": 71.78, "K": 0.06, "Ca": 8.75, "Ba": 0.0, "Fe": 0.0
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("predicted_class", response.json())


if __name__ == "__main__":
    unittest.main()