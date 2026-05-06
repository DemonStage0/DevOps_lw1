import os
import sys
import unittest

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from predict import GlassPredictor
from train import ModelTrainer


class TestGlassPredictor(unittest.TestCase):
    """Тесты для GlassPredictor."""

    @classmethod
    def setUpClass(cls) -> None:
        trainer = ModelTrainer()
        trainer.train()
        cls.predictor = GlassPredictor()

    def test_load_model(self):
        self.assertTrue(self.predictor.load_latest_model())

    def test_predict(self):
        features = [1.52101, 13.64, 4.49, 1.1, 71.78, 0.06, 8.75, 0.0, 0.0]
        prediction = self.predictor.predict(features)
        self.assertIn(prediction, [1, 2, 3, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()