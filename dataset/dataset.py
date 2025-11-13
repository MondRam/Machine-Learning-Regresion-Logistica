# dataset/dataset.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

print("✅ Todo instalado correctamente")

# ================================
# 1. ADQUISICIÓN DE DATOS
# ================================
df = pd.read_csv("dataset/bank-full.csv", sep=";")
print("✅ Dataset original cargado")

# ================================
# 2. LIMPIEZA DE DATOS
# ================================
print("Valores nulos por columna:\n", df.isnull().sum())
print("Duplicados encontrados:", df.duplicated().sum())
df = df.drop_duplicates()

# ================================
# 3. SEPARAR TARGET
# ================================
# Convertimos 'y' a 0/1
y = df["y"].map({"yes":1, "no":0})

# Variables independientes
X = df.drop(columns=["y"])

# ================================
# 4. TRANSFORMACIÓN DE DATOS
# ================================
# Codificación de variables categóricas
X_transformed = pd.get_dummies(X, drop_first=True)

# Normalización de columnas numéricas
scaler = MinMaxScaler()
X_transformed[["age", "balance"]] = scaler.fit_transform(X[["age", "balance"]])

# Guardar dataset minado
df_mined = pd.concat([X_transformed, y], axis=1)
os.makedirs("dataset", exist_ok=True)
df_mined.to_csv("dataset/bank-full-minado.csv", index=False)
print("✅ Archivo 'bank-full-minado.csv' generado con éxito.")

# ================================
# 5. ENTRENAMIENTO DE REGRESIÓN LOGÍSTICA
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Modelo de Regresión Logística entrenado")

# ================================
# 6. GUARDAR MODELO Y COLUMNAS
# ================================
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/regresion_logistica.pkl")
joblib.dump(list(X_transformed.columns), "model/columns.pkl")
print("✅ Modelo y columnas guardados en 'model/'")
