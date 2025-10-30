import os
import pandas as pd
import psycopg2
from config import DB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import json


# ----------------------------
# Cargar dataset limpio
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # El csv se carga desde la carpeta dataset
DATA_PATH = os.path.join(BASE_DIR, "dataset", "bank-full-minado.csv")
df = pd.read_csv(DATA_PATH)

# ----------------------------
# Leer nuevos datos desde la BD (si hay)
# ----------------------------
try:
    conn = psycopg2.connect(**DB)
    df_new = pd.read_sql("SELECT data, y_yes FROM new_data", conn) # En caso de que ya se hayan guardado previamente datos de entrenamiento
    conn.close()

    if not df_new.empty:
        df_new_expanded = pd.json_normalize(df_new['data']) #En caso de que haya más datos en la db se normalizan y se añaden al json
        df_new_expanded['y_yes'] = df_new['y_yes']
        df = pd.concat([df, df_new_expanded], ignore_index=True)
except Exception as e:
    print("No se pudieron cargar datos nuevos de la BD:", e)

# ----------------------------
# Separar variables
# ----------------------------
y = df["y_yes"] # Se marca la variable dependiente
X = df.drop(columns=["y_yes"]) #Se indican el resto de columnas como variables independientes al dropear la columna objetivo

# ----------------------------
# División de datos
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% entrenamiento 20% pruebas
