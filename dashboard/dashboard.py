# dashboard/dashboard.py

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import ttest_1samp
import json
import numpy as np

API_URL = "http://127.0.0.1:8000"  # Cambia si tu API está en otra URL

st.title("🤖 Dashboard - Modelo Logístico")

# -----------------------------
# 1️⃣ Formulario de inserción
# -----------------------------
st.header("🧾 Insertar nuevo registro")
with st.form("formulario"):
    age = st.number_input("Edad", 18, 100)
    job = st.selectbox("Ocupación", ["admin.","blue-collar","technician","services","management"])
    marital = st.selectbox("Estado civil", ["single","married","divorced"])
    education = st.selectbox("Educación", ["primary","secondary","tertiary"])
    balance = st.number_input("Balance", -5000, 100000)
    housing = st.selectbox("Hipoteca", ["yes","no"])
    loan = st.selectbox("Préstamo", ["yes","no"])
    y = st.selectbox("Aceptó producto", [0,1])
    submitted = st.form_submit_button("Guardar y reentrenar")

    if submitted:
        res = requests.post(f"{API_URL}/insertar_datos/", json={
            "age": age, "job": job, "marital": marital, "education": education,
            "balance": balance, "housing": housing, "loan": loan, "y": y
        })
        if res.ok:
            st.success("✅ Dato insertado y modelo reentrenado.")
        else:
            st.error(f"❌ Error al insertar: {res.text}")

# -----------------------------
# 2️⃣ Métricas históricas
# -----------------------------
st.header("📈 Métricas del modelo")
res = requests.get(f"{API_URL}/metricas/")

if res.ok:
    data = res.json()
    if data:
        df = pd.DataFrame(data)
        
        # Tabla histórica
        st.subheader("Tabla Histórica")
        st.dataframe(df)

        # Gráfica de métricas
        chart_df = df[["timestamp","accuracy","precision","recall","f1"]].set_index("timestamp")
        st.line_chart(chart_df)

        # Última matriz de confusión
        cm = df["matriz_confusion"].iloc[-1]
        if cm is not None:
            cm = np.array(cm)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No hay matriz de confusión disponible")

        # Curva Precision-Recall
        pr_precision = df["pr_precision"].iloc[-1]
        pr_recall = df["pr_recall"].iloc[-1]
        if pr_precision and pr_recall:
            fig, ax = plt.subplots()
            ax.plot(pr_recall, pr_precision, marker='.')
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Curva Precision-Recall")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No hay datos de Precision-Recall disponibles")

       