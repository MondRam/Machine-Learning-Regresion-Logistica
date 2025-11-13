# dashboard/dashboard.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import ttest_1samp
import json
import numpy as np
import os

API_URL = os.getenv("API_URL")  # e.g., https://your-railway-app.up.railway.app

st.title(" Dashboard - Modelo Logístico")

# -----------------------------
# 🔄 Insertar registro y predecir (formulario único)
# -----------------------------
st.header(" Insertar registro y obtener predicción")

with st.form("formulario_unico"):
    age = st.number_input("Edad", min_value=18, max_value=100, value=18)
    job = st.selectbox("Ocupación", ["admin.","blue-collar","technician","services","management"])
    marital = st.selectbox("Estado civil", ["single","married","divorced"])
    education = st.selectbox("Educación", ["primary","secondary","tertiary"])
    balance = st.number_input("Balance", min_value=-100000, max_value=1000000, value=-5000)
    housing = st.selectbox("Hipoteca", ["yes","no"])
    loan = st.selectbox("Préstamo", ["yes","no"])
    y = st.selectbox("Aceptó producto", [0, 1])

    submitted = st.form_submit_button("Guardar y predecir")

if submitted:
    if not API_URL:
        st.error(" API_URL no está definida en variables de entorno.")
    else:
        payload = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "y": y
        }

        # 1) Insertar y reentrenar
        try:
            res_insert = requests.post(f"{API_URL}/insertar_datos/", json=payload, timeout=30)
            if res_insert.ok:
                st.success(" Registro guardado y reentrenamiento disparado.")
            else:
                st.error(f" Error al insertar: {res_insert.status_code} {res_insert.text}")
        except Exception as e:
            st.error(f" Error de conexión al insertar: {e}")

        # 2) Pedir predicción
        try:
            res_pred = requests.post(f"{API_URL}/predecir/", json=payload, timeout=30)
            if res_pred.ok:
                resultado = res_pred.json()
                if "prediccion" in resultado:
                    st.success(f"🔮 Predicción: {resultado['prediccion']}")
                    probs = resultado.get("probabilidades")
                    if isinstance(probs, list):
                        st.write("Probabilidades:", probs)
                    else:
                        st.info("ℹ No se recibieron probabilidades.")
                elif "error" in resultado:
                    st.error(f" Error en predicción: {resultado['error']}")
                    trace = resultado.get("trace")
                    if trace:
                        with st.expander("Ver detalle técnico"):
                            st.code(trace)
                else:
                    st.warning(" Respuesta inesperada del servidor de predicción.")
            else:
                st.error(f" Error en predicción: {res_pred.status_code} {res_pred.text}")
        except Exception as e:
            st.error(f" Error de conexión al predecir: {e}")

# -----------------------------
#  Métricas del modelo (sin cambios)
# -----------------------------
st.header(" Métricas del modelo")

if not API_URL:
    st.error(" API_URL no está definida.")
else:
    try:
        res = requests.get(f"{API_URL}/metricas/", timeout=30)
        if res.ok:
            data = res.json()
            if data:
                df = pd.DataFrame(data)

                # Tabla histórica
                st.subheader("Tabla histórica")
                st.dataframe(df)

                # Gráfica de métricas
                if set(["timestamp","accuracy","precision","recall","f1"]).issubset(df.columns):
                    chart_df = df[["timestamp","accuracy","precision","recall","f1"]].set_index("timestamp")
                    st.line_chart(chart_df)
                else:
                    st.info(" Aún no hay suficientes métricas para graficar.")

                # Última matriz de confusión
                if "matriz_confusion" in df.columns and len(df) > 0:
                    cm = df["matriz_confusion"].iloc[-1]
                    if cm is not None:
                        cm = np.array(cm)
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay(cm).plot(ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning(" No hay matriz de confusión disponible.")
                else:
                    st.info(" No hay columna 'matriz_confusion' disponible.")

                # Curva Precision-Recall
                if "pr_precision" in df.columns and "pr_recall" in df.columns and len(df) > 0:
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
                        st.warning(" No hay datos de Precision-Recall disponibles.")
                else:
                    st.info(" No hay columnas de Precision-Recall disponibles.")

                # Prueba de hipótesis (accuracy > 0.9)
                if "accuracy" in df.columns:
                    try:
                        accuracy_vals = df["accuracy"].astype(float)
                        if len(accuracy_vals) > 0:
                            t_stat, p_val = ttest_1samp(accuracy_vals, 0.9)
                            alpha = 0.05
                            if p_val/2 < alpha and t_stat > 0:
                                st.success(" Rechazamos H0: el modelo ha mejorado significativamente.")
                            else:
                                st.warning(" No se puede rechazar H0.")
                        else:
                            st.info(" Aún no hay valores de accuracy para la prueba.")
                    except Exception:
                        st.info(" No fue posible calcular la prueba de hipótesis.")
            else:
                st.warning(" No hay métricas registradas aún.")
        else:
            st.error(f" Error al obtener métricas: {res.status_code} {res.text}")
    except Exception as e:
        st.error(f" Error al procesar métricas: {e}")