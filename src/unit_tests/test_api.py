import os
import sys
import shutil
import unittest
from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from app import app


class TestAPI(unittest.TestCase):
    """Тесты для API эндпоинтов."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        """Удаляем experiments перед каждым тестом для изоляции."""
        if os.path.exists("experiments"):
            shutil.rmtree("experiments")

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