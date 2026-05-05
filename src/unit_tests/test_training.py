import os
import sys
import unittest
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from train import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Тесты для ModelTrainer."""

    def setUp(self) -> None:
        """Подготовка перед каждым тестом."""
        self.trainer = ModelTrainer()

    def test_training(self):
        """Проверка успешности обучения и сохранения эксперимента."""
        result = self.trainer.train()
        self.assertEqual(result['status'], 'success')
        self.assertGreater(result['f1_score'], 0.5)
        self.assertTrue(os.path.exists(result['experiment_path']))
        self.assertTrue(
            os.path.isfile(
                os.path.join(result['experiment_path'], "trained_model.pkl")
            )
        )


if __name__ == "__main__":
    unittest.main()