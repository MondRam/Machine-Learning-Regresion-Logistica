# -*- coding: utf-8 -*-
import os
import pandas as pd
import psycopg2
from config import DB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
)
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Cargar dataset limpio
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "bank-full-minado.csv")
df = pd.read_csv(DATA_PATH)

# Leer nuevos datos desde la base de datos (si hay)
try:
    conn = psycopg2.connect(**DB)
    df_new = pd.read_sql("SELECT data, y_yes FROM new_data", conn)
    conn.close()

    if not df_new.empty:
        df_new_expanded = pd.json_normalize(df_new["data"])
        df_new_expanded["y_yes"] = df_new["y_yes"]
        df = pd.concat([df, df_new_expanded], ignore_index=True)
except Exception as e:
    print("No se pudieron cargar datos nuevos desde la base de datos:", e)

# Separar variables
y = df["y_yes"]
X = df.drop(columns=["y_yes"])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

metrics = {
    "modelo": "Regresión Logística",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc": auc,
    "matriz_confusion": cm.tolist()
}

print("Modelo entrenado correctamente")
print("Métricas obtenidas:", json.dumps(metrics, indent=4))

# Graficar curvas ROC y Precision-Recall
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend()
plt.show()

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.show()

# Guardar modelo y métricas en rutas absolutas
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "regresion_logistica.pkl")
metrics_path = os.path.join(MODEL_DIR, "metricas.json")

joblib.dump(model, model_path)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("Modelo guardado en:", model_path)
print("Métricas guardadas en:", metrics_path)

# Guardar métricas en la base de datos
try:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO model_metrics (timestamp, modelo, accuracy, precision, recall, f1, auc)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        datetime.now(),
        metrics["modelo"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc"]
    ))
    conn.commit()
    cur.close()
    conn.close()
    print("Métricas guardadas correctamente en la base de datos.")
except Exception as e:
    print("Error al guardar métricas en la base de datos:", e)
