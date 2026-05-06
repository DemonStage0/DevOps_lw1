import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from predict import GlassPredictor
from train import ModelTrainer
from logger import Logger

logger = Logger(True)
log = logger.get_logger(__name__)

app = FastAPI(
    title="Glass Classification API",
    description="API для классификации типов стекла по химическому составу",
    version="1.0.0"
)

predictor = GlassPredictor()


@app.get("/")
async def root():
    return {"message": "Glass Classification API is running"}


@app.get("/train")
async def train_model():
    """Запуск обучения модели."""
    try:
        trainer = ModelTrainer()
        result = trainer.train()
        if result['status'] == 'success':
            f1 = result['f1_score']
            log.info(f"Модель обучена. F1 = {f1:.4f}")
            return {
                "message": f"Модель обучена успешно. F1 = {f1:.4f}",
                "f1_score": f1
            }
        return {"message": "Модель не обучена"}
    except Exception as e:
        log.error(f"Ошибка обучения: {str(e)}")
        raise HTTPException(status_code=500, detail="Модель не обучена")


@app.get("/predict")
async def predict_class(
    RI: float = Query(1.52101, description="Показатель преломления"),
    Na: float = Query(13.64, description="Натрий"),
    Mg: float = Query(4.49, description="Магний"),
    Al: float = Query(1.1, description="Алюминий"),
    Si: float = Query(71.78, description="Кремний"),
    K: float = Query(0.06, description="Калий"),
    Ca: float = Query(8.75, description="Кальций"),
    Ba: float = Query(0.0, description="Барий"),
    Fe: float = Query(0.0, description="Железо")
):
    """Предсказание класса стекла по 9 признакам."""
    if not os.path.exists("experiments"):
        raise HTTPException(
            status_code=404,
            detail="Нет предобученной модели. Выполните /train"
        )

    features = [RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]
    prediction = predictor.predict(features)

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Нет предобученной модели. Выполните /train"
        )

    return {"predicted_class": prediction}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)