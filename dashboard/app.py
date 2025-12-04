import os
import html
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Asistente de Contratación Bancaria", layout="wide")

# Estilos globales (ligero y oscuro) y componentes visuales
st.markdown(
    """
    <style>
      .app-card { border:1px solid #e5e7eb; border-radius:12px; padding:16px; background: rgba(127,127,127,0.04); margin: 10px 0 16px; }
      @media (prefers-color-scheme: dark) {
        .app-card { border-color:#30363d; background: rgba(255,255,255,0.05); }
      }
      .bubble { border-radius:14px; padding:10px 12px; border:1px solid; max-width:100%; }
      .bubble.assistant { border-color:#94a3b8; background:rgba(148,163,184,0.15); }
      .bubble.user { border-color:#1f6feb; background:rgba(31,111,235,0.12); }
      .section-title { margin: 0 0 0.5rem 0; }
      .muted { color:#6b7280; font-size:0.9rem; }
      /* Chat nativo Streamlit: centrar y limitar ancho (un poco más ancho) */
      [data-testid="stChatInput"] > div { max-width: 900px; margin: 0 auto; }
      [data-testid="stChatMessage"] { max-width: 900px; margin-left: auto; margin-right: auto; }
      /* Reducir espacio superior para acercar el input al chat */
      section.main > div.block-container { padding-top: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_base() -> str:
    # Prefer variable de entorno; si no existe, intentar en secrets; por último localhost
    api = os.getenv("API_BASE_URL", None)
    if not api:
        try:
            api = st.secrets["API_BASE_URL"]
        except Exception:
            api = None
    if not api:
        api = "http://localhost:8000"
    return api.rstrip("/")


API_BASE = get_api_base()

# Mapas ES→EN para normalizar valores del formulario (disponibles en todo el módulo)
JOB_ES_TO_EN = {
    "Administración": "admin.",
    "Obrero": "blue-collar",
    "Emprendedor": "entrepreneur",
    "Empleada doméstica": "housemaid",
    "Management": "management",
    "Jubilado": "retired",
    "Autónomo": "self-employed",
    "Servicios": "services",
    "Estudiante": "student",
    "Técnico": "technician",
    "Desempleado": "unemployed",
    "Desconocido": "unknown",
}

MARITAL_ES_TO_EN = {
    "Soltero": "single",
    "Casado": "married",
    "Divorciado": "divorced",
    "Desconocido": "unknown",
}

EDU_ES_TO_EN = {
    "Primaria (4y)": "basic.4y",
    "Secundaria (6y)": "basic.6y",
    "Básica (9y)": "basic.9y",
    "Secundaria": "high.school",
    "Curso profesional": "professional.course",
    "Universidad": "university.degree",
    "Iletrado": "illiterate",
    "Desconocido": "unknown",
}

YES_NO_UNK_ES_TO_EN = {
    "Sí": "yes",
    "No": "no",
    "Desconocido": "unknown",
}

CONTACT_ES_TO_EN = {
    "Celular": "cellular",
    "Teléfono": "telephone",
}

MONTH_ES_TO_EN = {
    "ene": "jan",
    "feb": "feb",
    "mar": "mar",
    "abr": "apr",
    "may": "may",
    "jun": "jun",
    "jul": "jul",
    "ago": "aug",
    "sep": "sep",
    "oct": "oct",
    "nov": "nov",
    "dic": "dec",
}

WDAY_ES_TO_EN = {
    "Lun": "mon",
    "Mar": "tue",
    "Mié": "wed",
    "Jue": "thu",
    "Vie": "fri",
}


@st.cache_data(ttl=60)
def get_latest_model():
    try:
        r = requests.get(f"{API_BASE}/model/latest", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def get_metrics(limit: int = 10):
    try:
        r = requests.get(f"{API_BASE}/metrics", params={"limit": limit}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def predict_rl(features: dict):
    try:
        r = requests.post(f"{API_BASE}/predict_rl", json={"features": features}, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def predict_dl(features: dict):
    try:
        r = requests.post(f"{API_BASE}/predict_dl", json={"features": features}, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


## Reentrenamiento y feedback deshabilitados en UI
# def retrain(file_bytes: bytes | None):
#     files = None
#     if file_bytes:
#         files = {"labeled_csv": ("new_data.csv", file_bytes, "text/csv")}
#     try:
#         r = requests.post(f"{API_BASE}/retrain", files=files, timeout=600)
#         r.raise_for_status()
#         return r.json(), None
#     except Exception as e:
#         return None, str(e)

# def send_feedback(features: dict, y: int):
#     try:
#         r = requests.post(f"{API_BASE}/feedback", json={"features": features, "y": int(y)}, timeout=60)
#         r.raise_for_status()
#         return r.json(), None
#     except Exception as e:
#         return None, str(e)


# --- Chatbot helpers ---
YES_TOKENS = {"si", "sí", "yes", "y", "true"}
NO_TOKENS = {"no", "false"}


def _contains_any(text: str, words: set[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def parse_text_to_features(text: str) -> dict:
    """Parser simple en español/inglés para extraer features comunes.
    Cubre un subconjunto; el modelo imputará el resto.
    """
    import re

    t = text.lower()
    feats: dict = {}

    # Edad
    m = re.search(r"(?:edad|age)\s*[:=]?\s*(\d{1,3})", t)
    if m:
        feats["age"] = int(m.group(1))

    # Balance/Saldo
    m = re.search(r"(?:saldo|balance)\s*[:=]?\s*(-?\d+)", t)
    if m:
        feats["balance"] = int(m.group(1))

    # Estados binarios: housing, loan, default
    if _contains_any(t, {"hipoteca", "vivienda", "housing", "casa"}):
        feats["housing"] = "yes" if _contains_any(t, YES_TOKENS) else ("no" if _contains_any(t, NO_TOKENS) else "unknown")
    if _contains_any(t, {"prestamo", "crédito", "loan"}):
        feats["loan"] = "yes" if _contains_any(t, YES_TOKENS) else ("no" if _contains_any(t, NO_TOKENS) else "unknown")
    if _contains_any(t, {"default", "mora", "incumplimiento"}):
        feats["default"] = "yes" if _contains_any(t, YES_TOKENS) else ("no" if _contains_any(t, NO_TOKENS) else "unknown")

    # Estado civil
    if _contains_any(t, {"soltero", "single"}):
        feats["marital"] = "single"
    elif _contains_any(t, {"casado", "married"}):
        feats["marital"] = "married"
    elif _contains_any(t, {"divorciado", "divorced"}):
        feats["marital"] = "divorced"

    # Trabajo (mapa básico)
    job_map = {
        "administr": "admin.",
        "blue": "blue-collar",
        "empr": "entrepreneur",
        "house": "housemaid",
        "manage": "management",
        "retir": "retired",
        "self": "self-employed",
        "serv": "services",
        "student": "student",
        "tech": "technician",
        "unem": "unemployed",
    }
    for key, val in job_map.items():
        if key in t:
            feats["job"] = val
            break

    # Educación
    edu_map = {
        "univers": "university.degree",
        "secund": "high.school",
        "básic": "basic.9y",
        "basica": "basic.9y",
        "iliter": "illiterate",
        "profes": "professional.course",
    }
    for key, val in edu_map.items():
        if key in t:
            feats["education"] = val
            break

    # Contacto
    if _contains_any(t, {"celular", "móvil", "cellular", "mobile"}):
        feats["contact"] = "cellular"
    elif _contains_any(t, {"teléfono", "telefono", "telephone"}):
        feats["contact"] = "telephone"

    # Mes (abreviado en inglés)
    months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    for mname in months:
        if mname in t:
            feats["month"] = mname
            break

    # Día del mes (day)
    m = re.search(r"\b(?:day|día)\s*[:=]?\s*(\d{1,2})\b", t)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31:
            feats["day"] = day

    return feats


def section_header(title: str):
    st.markdown(f"### {title}")


def draw_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    st.pyplot(fig)


def draw_roc(roc_curve_data):
    fpr = roc_curve_data.get("fpr", [])
    tpr = roc_curve_data.get("tpr", [])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC")
    ax.legend()
    st.pyplot(fig)


def draw_pr(pr_curve_data):
    precision = pr_curve_data.get("precision", [])
    recall = pr_curve_data.get("recall", [])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(recall, precision, label="P-R")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend()
    st.pyplot(fig)


def draw_y_dist(y_dist):
    labels = list(y_dist.keys())
    values = list(y_dist.values())
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Distribución de y")
    st.pyplot(fig)


st.title("Asistente de Contratación Bancaria")

# Pestañas: inicio (walkthrough), chat, formulario y métricas
tab_home, tab_chat, tab_form, tab_metrics = st.tabs(["Inicio", "Chat", "Formulario", "Métricas"])

with tab_home:
    st.subheader("Guía rápida")
    st.markdown(
        """
        - Chat: escribe en lenguaje natural (por ejemplo: "Tengo 36 años, saldo 600, hipoteca sí, casado"). Respondo con “SÍ/NO” y un porcentaje.
        - Formulario: completa los campos en español. Puedes abrir el panel de parámetros avanzados si lo necesitas.
        - Métricas: consulta rendimiento (Exactitud, Precisión, Recall, F1, ROC-AUC), gráficas y el historial de versiones.
        - Reentrenado: si está activo, tras cada predicción se crea una nueva versión; usa “Actualizar métricas” para ver cambios.
        """
    )

    st.markdown("#### ¿Qué puedo preguntar en el Chat?")
    st.markdown(
        """
        - "Tengo 45 años, saldo 1200, hipoteca sí, casado, contacto celular. ¿Qué probabilidad tengo?"
        - "Soy soltero, 29 años, saldo 0, sin préstamos. ¿Me verías contratando?"
        - "Trabajo en management, 50 años, hipoteca sí, préstamo no. ¿Sería un sí o no?"
        - "Cliente 36 años, estudios universitarios, saldo 600, sin hipoteca, campaña anterior 2, mes jun. ¿Sí o no?"
        - "Estado civil: casado, profesión: services, contacto: telefónico, días desde último contacto 999. ¿Qué estimas?"
        - "Tengo 58 años, préstamo personal sí, hipoteca no, saldo 150. ¿Cuál es la probabilidad?"
        - "Profesión blue-collar, educación secundaria, 40 años, contacto celular, mes may, día 5. ¿Me aceptarías?"
        - "Soy admin, 35 años, sin préstamos, sin hipoteca, saldo 3000, campaña actual 1. ¿Resultado?"
        """
    )
    st.info("Tip: puedes dar pocos datos y el sistema completa faltantes de forma segura.")

    with st.expander("¿Cómo funciona el modelo y el minado de datos?", expanded=False):
        st.markdown(
            """
            - Entrenamiento sin dataset embebido: al iniciar la API, se mina el dataset desde una de dos fuentes configurables por entorno:
              - `TRAINING_SQL`: una consulta SQL que corre sobre `DATABASE_URL` y devuelve columnas de features y la etiqueta `y`.
              - `TRAINING_SOURCE_URL`: una URL que entrega CSV o JSON con las mismas columnas.
            - Limpieza y preparación: se normalizan categorías (p. ej. valores en español → formato del modelo), se descarta `duration` y se construye un pipeline con `OneHotEncoder` + `StandardScaler` + Regresión Logística.
            - Versionado: cada entrenamiento crea una versión de modelo con métricas (Accuracy, Precision, Recall, F1, ROC-AUC) que puedes ver en la pestaña **Métricas**.
            - Predicción: puedes escribir en el **Chat** o usar el **Formulario** en español; el backend completa faltantes de forma segura.
            - Reentrenamiento automático: si `AUTO_RETRAIN_AFTER_PREDICTION=true`, la API reentrena en background tras cada predicción usando de nuevo la fuente minada (y ejemplos etiquetados si hay).
            """
        )


with tab_chat:
    hdr_l, hdr_c, hdr_r = st.columns([0.7, 0.15, 0.15])
    with hdr_l:
        st.subheader("Asistente tipo Chat")
        st.markdown("<div class='muted'>Chatea en español y obtén una estimación clara.</div>", unsafe_allow_html=True)
    with hdr_c:
        use_dl_chat = st.toggle(
            "Usar Deep Learning",
            value=(st.session_state.get("backend_choice_chat", "RL") == "DL"),
            key="use_dl_chat",
        )
        st.session_state["backend_choice_chat"] = "DL" if use_dl_chat else "RL"
    with hdr_r:
        if st.button("Limpiar chat"):
            st.session_state.pop("messages", None)
            st.session_state.pop("last_features", None)
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hola, cuéntame algunos datos (edad, saldo, hipoteca sí/no, estado civil…) y estimo tu probabilidad."}
        ]

    # Render nativo de mensajes (centrado y ancho acotado por CSS)
    for m in st.session_state.messages:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            # Mostrar detalles técnicos si existen
            meta = m.get("meta")
            if role == "assistant" and meta:
                with st.expander("Ver más"):
                    st.markdown("**Endpoint**")
                    st.code(meta.get("endpoint", ""))
                    st.markdown("**JSON enviado**")
                    st.json(meta.get("request", {}))
                    st.markdown("**Respuesta de la API**")
                    st.json(meta.get("response", {}))

    # Input fijo al fondo del viewport, centrado por CSS
    user_input = st.chat_input("Escribe tu mensaje…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Parsear a features y predecir
        feats = parse_text_to_features(user_input)
        st.session_state.last_features = feats

        with st.spinner("Analizando y consultando la API…"):
            use_rl = st.session_state.get("backend_choice_chat", "RL") == "RL"
            if use_rl:
                resp, err = predict_rl(feats)
            else:
                resp, err = predict_dl(feats)
        if err:
            st.error(f"Error: {err}")
            reply = "Hubo un error al consultar la API."
        else:
            use_rl = st.session_state.get("backend_choice_chat", "RL") == "RL"
            prob = float(resp.get("probability", 0.0))
            pred = int(resp.get("predicted", 0))
            if feats:
                det = ", ".join([f"{k}={v}" for k, v in feats.items()])
                intro = f"Con lo que mencionaste ({det})"
            else:
                intro = "Con información parcial (faltan datos; completaré faltantes con valores por defecto)"

            tono = "Es probable que " + ("SÍ " if pred == 1 else "NO ") + "contrates"
            reply = f"{intro}, mi estimación es: {tono} (≈ {prob:.0%})."
            if 0.45 <= prob <= 0.55:
                reply += " Nota: la probabilidad está cerca del 50%, la confianza es moderada."

        meta = {
            "endpoint": f"{API_BASE}/{'predict_rl' if st.session_state.get('backend_choice_chat','RL')=='RL' else 'predict_dl'}",
            "request": {"features": feats, "backend": st.session_state.get("backend_choice_chat", "RL")},
            "response": resp or {},
        }
        st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
        st.rerun()

    # Feedback y reentrenamiento removidos de la UI para simplificar la experiencia


with tab_form:
    head_l, head_r = st.columns([0.85, 0.15])
    with head_l:
        st.subheader("Formulario de predicción")
        st.caption("Completa los datos clave. Puedes dejar campos vacíos: el modelo completará faltantes de forma segura.")
    with head_r:
        use_dl_form = st.toggle(
            "Usar Deep Learning",
            value=(st.session_state.get("backend_choice_form", "RL") == "DL"),
            key="use_dl_form",
        )
        st.session_state["backend_choice_form"] = "DL" if use_dl_form else "RL"

    col1, col2 = st.columns(2)

    # Básicos en español
    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=35, help="Ingresa tu edad en años")
        job_es = st.selectbox("Ocupación", list(JOB_ES_TO_EN.keys()), index=0, help="Selecciona la categoría que mejor te describa")
        marital_es = st.selectbox("Estado civil", list(MARITAL_ES_TO_EN.keys()), index=1, help="Elige tu estado civil")
        education_es = st.selectbox("Nivel educativo", list(EDU_ES_TO_EN.keys()), index=6, help="Selecciona tu nivel de estudios")
        default_es = st.selectbox("¿En mora (default)?", list(YES_NO_UNK_ES_TO_EN.keys()), index=1, help="Si alguna vez estuviste en incumplimiento")
        balance = st.number_input("Saldo en cuenta", value=0, step=100, help="Saldo promedio en tu cuenta (puede ser negativo)")
        housing_es = st.selectbox("¿Tienes hipoteca?", list(YES_NO_UNK_ES_TO_EN.keys()), index=1)

    with col2:
        loan_es = st.selectbox("¿Tienes préstamo personal?", list(YES_NO_UNK_ES_TO_EN.keys()), index=1)
        contact_es = st.selectbox("Medio de contacto", list(CONTACT_ES_TO_EN.keys()), index=0, help="El canal por el que te contactamos")
        day = st.number_input("Día del mes", min_value=1, max_value=31, value=15, help="Día del contacto")
        month_es = st.selectbox("Mes", ["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"], index=7)
        day_of_week_es = st.selectbox("Día de la semana", ["Lun","Mar","Mié","Jue","Vie"], index=2)
        campaign = st.number_input("Número de contactos (campaña)", min_value=1, max_value=60, value=1, help="Cuántas veces te hemos contactado en esta campaña")

    # Avanzados ocultos para no confundir
    with st.expander("Opcional: ajustes avanzados (puedes ignorarlos)"):
        c3a, c3b = st.columns(2)
        with c3a:
            pdays = st.number_input("Días desde contacto previo (pdays)", min_value=-1, max_value=999, value=999, help="-1 si no hay registro previo")
            previous = st.number_input("Contactos previos", min_value=0, max_value=100, value=0)
            poutcome_es = st.selectbox("Resultado campaña previa", ["Fracaso","Inexistente","Éxito"], index=1)
        with c3b:
            emp_var_rate = st.number_input("Tasa variación empleo (emp.var.rate)", value=1.1, format="%0.2f")
            cons_price_idx = st.number_input("Índice precios (cons.price.idx)", value=93.75, format="%0.2f")
            cons_conf_idx = st.number_input("Índice confianza (cons.conf.idx)", value=-40.0, format="%0.1f")
            euribor3m = st.number_input("Euribor 3m", value=4.0, format="%0.3f")
            nr_employed = st.number_input("Empleados (nr.employed)", value=5191.0, format="%0.1f")

    # Mapear valores seleccionados en español a lo que espera el backend
    job = JOB_ES_TO_EN[job_es]
    marital = MARITAL_ES_TO_EN[marital_es]
    education = EDU_ES_TO_EN[education_es]
    default = YES_NO_UNK_ES_TO_EN[default_es]
    housing = YES_NO_UNK_ES_TO_EN[housing_es]
    loan = YES_NO_UNK_ES_TO_EN[loan_es]
    contact = CONTACT_ES_TO_EN[contact_es]
    month = MONTH_ES_TO_EN[month_es]
    day_of_week = WDAY_ES_TO_EN[day_of_week_es]
    poutcome = {"Fracaso": "failure", "Inexistente": "nonexistent", "Éxito": "success"}[poutcome_es]

    features = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": int(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": int(day),
        "month": month,
        "day_of_week": day_of_week,
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": poutcome,
        "emp.var.rate": float(emp_var_rate),
        "cons.price.idx": float(cons_price_idx),
        "cons.conf.idx": float(cons_conf_idx),
        "euribor3m": float(euribor3m),
        "nr.employed": float(nr_employed),
    }

    if st.button("Calcular probabilidad", type="primary"):
        with st.spinner("Consultando a la API..."):
            use_rl = st.session_state.get("backend_choice_form", "RL") == "RL"
            if use_rl:
                resp, err = predict_rl(features)
            else:
                resp, err = predict_dl(features)
        if err:
            st.error(f"Error en la predicción: {err}")
        elif resp:
            use_rl = st.session_state.get("backend_choice_form", "RL") == "RL"
            prob = float(resp.get("probability", 0.0))
            pred = int(resp.get("predicted", 0))
            nombre = "Clásico" if use_rl else "Deep Learning"
            if pred == 1:
                st.success(f"{nombre}: SÍ (≈ {prob:.0%}).")
            else:
                st.info(f"{nombre}: NO (≈ {prob:.0%}).")
            st.caption(f"Fecha: {str(resp.get('timestamp'))}")

            with st.expander("Ver más"):
                st.markdown("**Endpoint**")
                st.code(f"{API_BASE}/{'predict_rl' if use_rl else 'predict_dl'}")
                st.markdown("**JSON enviado**")
                st.json({"features": features, "backend": st.session_state.get("backend_choice_form", "RL")})
                st.markdown("**Respuesta de la API**")
                st.json(resp)

    # Fin formulario


with tab_metrics:
    st.subheader("Métricas del modelo y gráficas")
    # Botón para refrescar y limpiar caché (evita esperar TTL)
    if st.button("Actualizar métricas", help="Limpia caché y vuelve a consultar la API"):
        st.cache_data.clear()
        st.rerun()

    data = get_metrics(limit=10)
    if "error" in data:
        st.warning("No se pudieron obtener métricas de la API. Mostrando UI.")
    else:
        history = data.get("history", [])
        if not history:
            st.info("Aún no hay métricas disponibles.")
        else:
            latest = history[0]
            m = latest.get("metrics", {})

            # Mostrar versión actual y conteo de versiones disponibles
            lm = get_latest_model()
            if "error" not in lm:
                st.caption(f"Versión actual: {lm.get('version')} · Modelos guardados: {len(history)}")

            st.markdown("##### Resumen rápido")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Exactitud (Accuracy)", f"{m.get('accuracy', 0):.3f}")
            c2.metric("Precisión (Sí)", f"{m.get('precision', 0):.3f}")
            c3.metric("Cobertura/Recall (Sí)", f"{m.get('recall', 0):.3f}")
            c4.metric("F1 (Sí)", f"{m.get('f1', 0):.3f}")
            c5.metric("ROC-AUC", f"{m.get('roc_auc', 0):.3f}")

            with st.expander("¿Qué significa cada métrica?"):
                st.markdown(
                    "- Exactitud: porcentaje de aciertos globales.\n"
                    "- Precisión (Sí): de los casos predichos como Sí, cuántos realmente eran Sí.\n"
                    "- Cobertura/Recall (Sí): de todos los Sí reales, cuántos detecta.\n"
                    "- F1: equilibrio entre Precisión y Recall.\n"
                    "- ROC-AUC: capacidad del modelo para separar Sí/No (más alto es mejor)."
                )

            with st.expander("¿Cómo interpretar las gráficas?"):
                st.markdown(
                    "- Matriz de confusión: muestra aciertos y errores por clase (TN, FP, FN, TP).\n"
                    "- Curva ROC: relación TPR vs FPR; la diagonal es azar, cuanto más arriba/mejor.\n"
                    "- Curva Precisión-Recall: útil con clases desbalanceadas; observa el área (PR-AUC).\n"
                    "- Distribución de y: balance de clases (0=No, 1=Sí); fuerte desequilibrio puede afectar métricas."
                )

            st.markdown("---")
            # Gráficas
            gc1, gc2, gc3, gc4 = st.columns(4)
            with gc1:
                draw_confusion_matrix(m.get("confusion_matrix", [[0,0],[0,0]]))
            with gc2:
                draw_roc(m.get("roc_curve", {}))
            with gc3:
                draw_pr(m.get("pr_curve", {}))
            with gc4:
                draw_y_dist(m.get("y_distribution", {"0":0, "1":0}))
                st.caption("Objetivo (y): 0 = No, 1 = Sí")

            # Historial de reentrenamiento: tabla y gráficas
            with st.expander("¿Qué muestra el historial por versión?"):
                st.markdown(
                    "- Tabla histórica: cada fila corresponde a una versión entrenada con su timestamp.\n"
                    "- Incluye Accuracy/Precision/Recall/F1 y la matriz de confusión compacta.\n"
                    "- Úsalo para comparar rendimientos entre modelos y detectar regresiones."
                )
            st.markdown("#### Tabla histórica")
            hist_rows = []
            for h in history:
                m_h = h.get("metrics", {})
                cm = m_h.get("confusion_matrix", [[0, 0], [0, 0]])
                cm_str = f"{cm[0][0]},{cm[0][1]} · {cm[1][0]},{cm[1][1]}"
                tipo = m_h.get("model_type")
                modelo = "Deep Learning" if str(tipo).lower() in {"mlp", "torch", "pytorch"} else "Regresión Logística"
                hist_rows.append({
                    "timestamp": h.get("created_at"),
                    "modelo": modelo,
                    "version": h.get("version"),
                    "accuracy": m_h.get("accuracy"),
                    "precision": m_h.get("precision"),
                    "recall": m_h.get("recall"),
                    "f1": m_h.get("f1"),
                    "matriz_confusion": cm_str,
                })
            hist_df = pd.DataFrame(hist_rows)
            if hist_df.empty:
                st.info("Aún no hay historial. Envía una predicción y pulsa ‘Actualizar métricas’.")
            else:
                tab_all, tab_rl, tab_dl = st.tabs(["Todos", "RL", "DL"])
                with tab_all:
                    st.dataframe(hist_df, use_container_width=True)
                with tab_rl:
                    st.dataframe(hist_df[hist_df["modelo"] == "Regresión Logística"], use_container_width=True)
                with tab_dl:
                    st.dataframe(hist_df[hist_df["modelo"] == "Deep Learning"], use_container_width=True)

                with st.expander("¿Cómo leer el progreso por reentrenado?"):
                    st.markdown(
                        "- La línea temporal muestra cómo cambian Accuracy/Precision/Recall/F1 con cada versión.\n"
                        "- Si el timestamp no es parseable, se usa el índice por versión como alternativa.\n"
                        "- Un aumento sostenido sugiere mejora del modelo; caídas indican revisar datos o parámetros.\n"
                        "- Pulsa ‘Actualizar métricas’ tras nuevas predicciones si el reentrenado está activo."
                    )
                st.markdown("#### Progreso por reentrenado")
                # Gráfica temporal (por timestamp) y alternativa por versión
                time_df = hist_df[["timestamp", "accuracy", "precision", "recall", "f1"]].copy()
                time_df = time_df.set_index("timestamp")
                try:
                    st.line_chart(time_df)
                except Exception:
                    # Si algún backend devuelve timestamp no parseable, caer a índice por versión
                    st.line_chart(hist_df.set_index("version")[ ["accuracy","precision","recall","f1"] ])


## Pestaña de reentrenamiento eliminada para simplificar la interfaz

st.sidebar.info("API base: " + API_BASE)
st.sidebar.caption("Configura API_BASE_URL en Secrets o variables de entorno.")