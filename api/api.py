from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import text
from .config import DB
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import os
from datetime import datetime
import traceback
import json

app = FastAPI(title="API de Reentrenamiento - Regresión Logística")

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "regresion_logistica.pkl")

# ----------------------------
# Modelo Pydantic
# ----------------------------
class DatosEntrada(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    balance: float
    housing: str
    loan: str
    y: int

# ----------------------------
# Función para reentrenar modelo
# ----------------------------
def retrain_model():
    try:
        df = pd.read_sql(text("SELECT * FROM insertar_datos"), DB.connect())

        if df.empty:
            print("⚠️ No hay datos para reentrenar.")
            return

        X = df.drop(columns=["y"])
        y = df["y"]

        if len(y.unique()) < 2:
            print(f"⚠️ No se puede reentrenar: solo hay una clase ({y.unique()})")
            return

        X_encoded = pd.get_dummies(X, columns=["job", "marital", "education", "housing", "loan"], drop_first=True)

        columns_path = os.path.join(BASE_DIR, "model", "columns.pkl")
        if os.path.exists(columns_path):
            saved_columns = joblib.load(columns_path)
            for col in saved_columns:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            X_encoded = X_encoded[saved_columns]
        else:
            joblib.dump(X_encoded.columns, columns_path)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_encoded, y)

        joblib.dump(model, MODEL_PATH)

        y_pred = model.predict(X_encoded)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # Extra: matriz de confusión y curva PR
        matriz_confusion = confusion_matrix(y, y_pred).tolist()
        pr_precision, pr_recall, _ = precision_recall_curve(y, model.predict_proba(X_encoded)[:, 1])

        with DB.begin() as conn:
            conn.execute(
                text("""
                     INSERT INTO metricas (timestamp, modelo, accuracy, precision, recall, f1, matriz_confusion, pr_precision, pr_recall)
                     VALUES (:timestamp, :modelo, :acc, :prec, :rec, :f1, :matriz_confusion, :pr_precision, :pr_recall)
                     """),
                {
                    "timestamp": datetime.now(),
                    "modelo": "Regresión Logística",
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1,
                    "matriz_confusion": json.dumps(matriz_confusion),
                    "pr_precision": json.dumps(pr_precision.tolist()),
                    "pr_recall": json.dumps(pr_recall.tolist())
                }
            )
        print("✅ Métricas guardadas correctamente en metricas")

    except Exception as e:
        print("⚠️ Error al reentrenar el modelo:", e)
        print(traceback.format_exc())

# ----------------------------
# Endpoint para insertar datos y reentrenar
# ----------------------------
@app.post("/insertar_datos/")
def insertar_datos(data: DatosEntrada, background_tasks: BackgroundTasks):
    try:
        with DB.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO insertar_datos (age, job, marital, education, balance, housing, loan, y)
                    VALUES (:age, :job, :marital, :education, :balance, :housing, :loan, :y)
                """),
                data.model_dump()
            )
        background_tasks.add_task(retrain_model)
        return {"message": "✅ Datos insertados y reentrenamiento iniciado correctamente."}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# ----------------------------
# Endpoint para ver métricas
# ----------------------------
@app.get("/metricas/")
def get_metrics():
    df = pd.read_sql(text("SELECT * FROM metricas"), DB.connect())
    if not df.empty:
        df['timestamp'] = df['timestamp'].astype(str)
        for col in ['matriz_confusion', 'pr_precision', 'pr_recall']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and x not in [None, ""] else None)
        return df.to_dict(orient="records")
    return []

@app.get("/")
def home():
    return {
        "message": "✅ API de Reentrenamiento corriendo correctamente!",
        "endpoints": {
            "POST /insertar_datos/": "Inserta datos y reentrena el modelo",
            "GET /metricas/": "Obtiene las últimas métricas del modelo"
        }
    }
@app.post("/predecir/")
def predecir(data: DatosEntrada):
    try:
        # Cargar columnas y modelo
        columns_path = os.path.join(BASE_DIR, "model", "columns.pkl")
        modelo = joblib.load(MODEL_PATH)
        columnas = joblib.load(columns_path)

        # Convertir entrada a DataFrame y alinear columnas
        entrada = pd.DataFrame([data.model_dump()])
        entrada_encoded = pd.get_dummies(entrada, columns=["job", "marital", "education", "housing", "loan"], drop_first=True)

        for col in columnas:
            if col not in entrada_encoded.columns:
                entrada_encoded[col] = 0
        entrada_encoded = entrada_encoded[columnas]

        # Predicción
        pred = modelo.predict(entrada_encoded)[0]
        prob = modelo.predict_proba(entrada_encoded)[0].tolist()

        return {"prediccion": int(pred), "probabilidades": prob}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
