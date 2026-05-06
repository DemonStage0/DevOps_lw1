# DevOps_lw1 - Glass Classification ML Pipeline

## Описание проекта
Проект реализует классический жизненный цикл разработки ML модели для классификации типов стекла с использованием CI/CD пайплайна. Модель классифицирует стекло по 7 классам на основе 9 химических признаков.

## Структура проекта
```
DevOps_lw1/
├── CI/
│ └── Jenkinsfile
├── CD/
│ └── Jenkinsfile
├── data/
│ └── glass.csv
├── src/
│ ├── unit_tests/
│ │ ├── test_api.py
│ │ ├── test_predict.py
│ │ ├── test_preprocess.py
│ │ └── test_training.py
│ ├── app.py
│ ├── logger.py
│ ├── predict.py
│ ├── preprocess.py
│ └── train.py
├── tests/
│ ├── test_0.json
│ └── test_1.json
├── .gitignore
├── config.ini
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
└── requirements_freeze.txt
```

## Установка и запуск

### Локальный запуск
```
pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker запуск
```
docker-compose build
docker-compose up -d
```

## API Endpoints

### Health check
```
GET /
Ответ: {"message": "Glass Classification API is running"}
```

### Обучение модели
```
GET /train
Ответ: {"message": "Модель обучена успешно. F1 = 0.8562"}
```

### Предсказание класса стекла
```
GET /predict?RI=1.52101&Na=13.64&Mg=4.49&Al=1.1&Si=71.78&K=0.06&Ca=8.75&Ba=0.0&Fe=0.0
Ответ: {"predicted_class": 1}
```

### Классы стекла
```
1 - building_windows_float_processed
2 - building_windows_non_float_processed
3 - vehicle_windows_float_processed
5 - containers
6 - tableware
7 - headlamps
```

## Тестирование

```
python -m pytest src/unit_tests/ -v
```

## CI/CD Pipeline

### CI Pipeline (Jenkins)
- Клонирование репозитория из GitHub
- Сборка Docker образа с моделью
- Запуск unit-тестов внутри контейнера
- Публикация образа в Docker Hub

### CD Pipeline (Jenkins)
- Загрузка образа из Docker Hub
- Запуск контейнера с моделью
- Функциональное тестирование эндпоинтов /, /train и /predict
- Проверка корректности предсказаний

## Технологии
- Python 3.9 + scikit-learn (Random Forest)
- FastAPI + Uvicorn
- Docker + Docker Compose
- Jenkins (CI/CD)
- Git + GitHub

## Эксперименты
Каждый эксперимент сохраняется в experiments/exp_N/ и содержит:
- config.yml - параметры модели и хэш
- trained_model.pkl - сериализованная модель
- metrics.yml - метрики качества
- logs.txt - логи обучения