import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from datetime import datetime
from sqlalchemy import text
from api.config import DB
import json

DATA_PATH = "dataset/bank-full-minado.csv"
df = pd.read_csv(DATA_PATH)

if "y" not in df.columns:
    raise ValueError("❌ El dataset debe tener la columna 'y'")

X = df.drop(columns=["y"])
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = {
    "timestamp": datetime.now(),
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "matriz_confusion": confusion_matrix(y_test, y_pred).tolist()
}

print("✅ Modelo guardado correctamente")
print("✅ Métricas calculadas:", metrics)

joblib.dump(model, "model/regresion_logistica.pkl")

try:
    metrics_to_save = metrics.copy()
    metrics_to_save["matriz_confusion"] = json.dumps(metrics["matriz_confusion"])
    with DB.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO dbo.metricas (timestamp, accuracy, precision, recall, f1, matriz_confusion)
                VALUES (:timestamp, :accuracy, :precision, :recall, :f1, :matriz_confusion)
            """),
            metrics_to_save
        )
    print("✅ Métricas guardadas en dbo.metricas correctamente")
except Exception as e:
    print("❌ Error al guardar métricas en la BD:", e)
