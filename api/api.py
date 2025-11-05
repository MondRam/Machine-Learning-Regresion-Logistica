import os
from fastapi import FastAPI
import pandas as pd
import joblib
import json
import psycopg2
from datetime import datetime
from config import DB

app = FastAPI(title="API - Regresión Lineal Bank Marketing")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "linear_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

# Cargar modelo
model = joblib.load(MODEL_PATH)

# Cargar métricas
with open(METRICS_PATH) as f:
    metrics = json.load(f)


@app.get("/")
def root():
    return {"status": "API funcionando correctamente"}

@app.get("/metrics")
def get_metrics():
    try:
        # Guardar métricas en la BD
        conn = psycopg2.connect(**DB)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO model_metrics (timestamp, modelo, r2, mse, rmse, mae)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            datetime.now(),
            metrics["modelo"],
            metrics["R2"],
            metrics["MSE"],
            metrics["RMSE"],
            metrics["MAE"]
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return {"error": f"No se pudieron guardar métricas en la BD: {e}"}

    return metrics

