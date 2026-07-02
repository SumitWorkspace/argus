# Credit Card Fraud Detection Pipeline

## Project Overview
This project is an end-to-end Machine Learning pipeline for credit card fraud detection. It trains an XGBoost classifier on the Kaggle Credit Card Fraud dataset, serves predictions via a FastAPI endpoint, detects data drift in production requests, and packages the entire system inside Docker.

## How to Train the Model
Run the training script to load the data, scale numeric columns, train the XGBoost classifier, and generate the model and scaler artifacts:
```bash
python train.py
```

## How to Run with Docker
Use Docker Compose to build the image and spin up the container:
```bash
docker-compose up --build
```

## How to Test
After starting the application, test the endpoints using the interactive Swagger UI:
- Open your browser and visit: http://localhost:8000/docs

To view the live operations monitoring dashboard:
- Open your browser and visit: http://localhost:8000/dashboard
- Alternatively, you can open the static `dashboard.html` file directly.

## API Endpoints
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Health check endpoint returning model configuration and status. |
| `/predict` | `POST` | Accepts transaction features, scales them, and returns a fraud prediction, probability, and risk level. |
| `/drift` | `GET` | Computes statistical drift of recent predictions compared to baseline training distributions. |
| `/stats` | `GET` | Returns production traffic aggregates (total predictions, recent fraud rate, and timestamp info). |
| `/dashboard` | `GET` | Serves the web-based monitoring dashboard UI. |
