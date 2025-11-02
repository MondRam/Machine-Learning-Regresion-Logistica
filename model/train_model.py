# -*- coding: utf-8 -*-
import os
import pandas as pd
import psycopg2
# from config import DB
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

# ----------------------------
# Cargar dataset limpio
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # El csv se carga desde la carpeta dataset
DATA_PATH = os.path.join(BASE_DIR, "dataset", "bank-full-minado.csv")
df = pd.read_csv(DATA_PATH)

# ----------------------------
# Leer nuevos datos desde la BD (si hay)
# ----------------------------
# try:
#     conn = psycopg2.connect(**DB)
#     df_new = pd.read_sql("SELECT data, y_yes FROM new_data", conn)
#     conn.close()

#     if not df_new.empty:
#         df_new_expanded = pd.json_normalize(df_new['data'])
#         df_new_expanded['y_yes'] = df_new['y_yes']
#         df = pd.concat([df, df_new_expanded], ignore_index=True)
# except Exception as e:
#     print("No se pudieron cargar datos nuevos de la BD:", e)

# ----------------------------
# Separar variables
# ----------------------------
y = df["y_yes"]  # Variable dependiente
X = df.drop(columns=["y_yes"])  # Variables independientes

# ----------------------------
# División de datos
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Entrenar modelo
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# Evaluar (clasificación)
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidades para curva ROC y PR

# Métricas principales
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

metrics = {
    "modelo": "Regresión Logística",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1,
    "AUC": auc,
    "Matriz_Confusion": cm.tolist()  # Convertir a lista para guardar en JSON
}

print("\nModelo entrenado correctamente\n")
print(json.dumps(metrics, indent=4))

# ----------------------------
# Graficar curvas
# ----------------------------
# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend()
plt.show()

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.show()

# ----------------------------
# Guardar modelo y métricas en rutas absolutas
# ----------------------------
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "logistic_model.pkl")
metrics_path = os.path.join(MODEL_DIR, "metrics.json")

joblib.dump(model, model_path)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nModelo guardado en: {model_path}")
print(f"Métricas guardadas en: {metrics_path}")

# ----------------------------
# Guardar métricas en la BD
# ----------------------------
try:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO model_metrics (timestamp, modelo, accuracy, precision, recall, f1, auc)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        datetime.now(),
        metrics["modelo"],
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1-Score"],
        metrics["AUC"]
    ))
    conn.commit()
    cur.close()
    conn.close()
    print("\nMétricas guardadas en la base de datos correctamente.")
except Exception as e:
    print("Error al guardar métricas en la BD:", e)
