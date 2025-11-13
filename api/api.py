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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "regresion_logistica.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model", "columns.pkl")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "bank-full-minado.csv")  # Ruta corregida

class DatosEntrada(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    balance: float
    housing: str
    loan: str
    y: int

def crear_tablas_si_no_existen():
    try:
        with DB.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS insertar_datos (
                    age INT,
                    job TEXT,
                    marital TEXT,
                    education TEXT,
                    balance FLOAT,
                    housing TEXT,
                    loan TEXT,
                    y INT
                );
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS metricas (
                    timestamp TIMESTAMP,
                    modelo TEXT,
                    accuracy FLOAT,
                    precision FLOAT,
                    recall FLOAT,
                    f1 FLOAT,
                    matriz_confusion TEXT,
                    pr_precision TEXT,
                    pr_recall TEXT
                );
            """))
        print("✅ Tablas creadas o ya existentes.")
    except Exception as e:
        print("⚠️ Error al crear tablas:", e)
        print(traceback.format_exc())

# ----------------------------
# Cargar CSV inicial si la tabla está vacía
# ----------------------------
def cargar_csv_inicial():
    try:
        with DB.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM insertar_datos")).scalar()
        if count > 0:
            print("ℹ️ Datos ya existentes en insertar_datos. No se carga CSV.")
            return

        if not os.path.exists(CSV_PATH):
            print("⚠️ CSV no encontrado:", CSV_PATH)
            return

        df = pd.read_csv(CSV_PATH, sep=",")

        # ⚠️ Solo los primeros 1000 registros
        df = df.head(1000)

        # Reconstruir columnas categóricas desde las dummies
        job_cols = [c for c in df.columns if c.startswith("job_")]
        df["job"] = df[job_cols].idxmax(axis=1).str.replace("job_", "")

        marital_cols = [c for c in df.columns if c.startswith("marital_")]
        df["marital"] = df[marital_cols].idxmax(axis=1).str.replace("marital_", "")

        edu_cols = [c for c in df.columns if c.startswith("education_")]
        df["education"] = df[edu_cols].idxmax(axis=1).str.replace("education_", "")

        df["housing"] = df["housing_yes"].map({True: "yes", False: "no"})
        df["loan"] = df["loan_yes"].map({True: "yes", False: "no"})

        # Seleccionar columnas crudas + target
        df_final = df[["age", "job", "marital", "education", "balance", "housing", "loan", "y"]]

        # Inserción masiva
        with DB.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO insertar_datos (age, job, marital, education, balance, housing, loan, y)
                    VALUES (:age, :job, :marital, :education, :balance, :housing, :loan, :y)
                """),
                df_final.to_dict(orient="records")
            )
        print("✅ Se insertaron los primeros 1000 registros del CSV.")
    except Exception as e:
        print("⚠️ Error al cargar CSV:", e)
        print(traceback.format_exc())

# ----------------------------
# Reentrenamiento del modelo
# ----------------------------
def retrain_model():
    try:
        df = pd.read_sql(text("SELECT * FROM insertar_datos"), DB.connect())
        if df.empty:
            return

        X = df.drop(columns=["y"])
        y = df["y"]
        if len(y.unique()) < 2:
            return

        X_encoded = pd.get_dummies(X, columns=["job", "marital", "education", "housing", "loan"], drop_first=True)
        joblib.dump(X_encoded.columns.tolist(), COLUMNS_PATH)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_encoded, y)
        joblib.dump(model, MODEL_PATH)

        y_pred = model.predict(X_encoded)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        matriz_confusion = confusion_matrix(y, y_pred).tolist()
        pr_precision, pr_recall, _ = precision_recall_curve(y, model.predict_proba(X_encoded)[:, 1])

        with DB.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO metricas (
                        timestamp, modelo, accuracy, precision, recall, f1,
                        matriz_confusion, pr_precision, pr_recall
                    )
                    VALUES (
                        :timestamp, :modelo, :acc, :prec, :rec, :f1,
                        :matriz_confusion, :pr_precision, :pr_recall
                    )
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
    except Exception as e:
        print("Error en reentrenamiento:", e)
        print(traceback.format_exc())

# ----------------------------
# Endpoints
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
        return {"message": "Datos insertados y reentrenamiento iniciado."}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.post("/predecir/")
def predecir(data: DatosEntrada):
    try:
        if not os.path.exists(MODEL_PATH):
            return {"error": "No hay modelo entrenado aún."}

        model = joblib.load(MODEL_PATH)
        columnas = joblib.load(COLUMNS_PATH)

        entrada = pd.DataFrame([data.model_dump()])

        # Eliminar columnas irrelevantes como 'y' o 'id' si existen
        entrada = entrada.drop(columns=[col for col in ["y", "id"] if col in entrada.columns])

        entrada_encoded = pd.get_dummies(entrada, columns=["job", "marital", "education", "housing", "loan"], drop_first=True)

        # Asegurar que todas las columnas necesarias estén presentes
        for col in columnas:
            if col not in entrada_encoded.columns:
                entrada_encoded[col] = 0

        # Eliminar columnas sobrantes y ordenar
        entrada_encoded = entrada_encoded[[col for col in columnas]]

        pred = int(model.predict(entrada_encoded)[0])
        prob = model.predict_proba(entrada_encoded)[0].tolist()

        return {"prediccion": pred, "probabilidades": prob}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/metricas/")
def get_metrics():
    try:
        df = pd.read_sql(text("SELECT * FROM metricas ORDER BY timestamp"), DB.connect())
        if not df.empty:
            df['timestamp'] = df['timestamp'].astype(str)
            for col in ['matriz_confusion', 'pr_precision', 'pr_recall']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x else None)
            return df.to_dict(orient="records")
        return []
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/")
def home():
    return {
        "message": "API de Reentrenamiento corriendo",
        "endpoints": {
            "POST /insertar_datos/": "Inserta datos y reentrena el modelo",
            "GET /metricas/": "Obtiene métricas del modelo",
            "POST /predecir/": "Predice con el último modelo entrenado"
        }
    }

# ----------------------------
# Cargar CSV al iniciar
# ----------------------------
crear_tablas_si_no_existen()
cargar_csv_inicial()