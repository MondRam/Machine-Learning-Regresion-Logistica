# Sistema ML Completo: Regresión Logística (Bank Marketing)

Este proyecto implementa un sistema completo de ML para predecir si un cliente contratará un producto bancario. En esta versión no se incluye dataset embebido: el modelo se entrena a partir de un paso de minado de datos configurable (SQL/HTTP). Incluye:

- API REST con FastAPI para predicción y métricas.
- Modelo de Regresión Logística con `scikit-learn`, pipeline y métricas almacenadas.
- Base de datos PostgreSQL para predicciones, métricas y versiones del modelo.
- Dashboard web en Streamlit para ingresar datos (Formulario en español), modo Chat asistente y ver métricas.
 - Auto-reentrenamiento en background tras cada predicción (configurable) y versionado (`v1`, `v2`, ...).
 - En Chat y Formulario se incluye un panel "Ver más" con el endpoint usado, el JSON enviado y la respuesta recibida.
 - Botón "Actualizar métricas" en la pestaña Métricas para refrescar cache y mostrar la última versión.
- Listo para despliegue gratuito: API en Render, BD en Neon, Dashboard en Streamlit Community Cloud.

## Estructura

```
.
├── api/
│   ├── main.py          # Entrypoint FastAPI
│   ├── db.py            # Conexión SQLAlchemy
│   ├── models.py        # Tablas: model_versions, predictions, labeled_examples
│   ├── crud.py          # Operaciones DB
│   ├── schemas.py       # Pydantic schemas
│   └── ml.py            # Entrenamiento, métricas y serialización
├── dashboard/
│   └── app.py           # App Streamlit (UI)
├── requirements.txt
├── render.yaml          # Config para Render (API)
└── README.md
```

## Requerimientos

- Python 3.10+
- Postgres (local o en la nube). Para desarrollo local, por defecto usa SQLite (`local.db`). Para producción, usar `DATABASE_URL` (Postgres).

## Variables de Entorno y Secrets

- Crea un archivo `.env` (no se versiona) a partir de `.env.example` y define al menos:
  - `DATABASE_URL` → cadena de conexión Postgres (por ejemplo, Neon) con `sslmode=require`.
  - `TRAINING_SQL` (opcional) → consulta SQL a ejecutar sobre `DATABASE_URL` que devuelva columnas de features y `y`.
  - `TRAINING_SOURCE_URL` (opcional) → URL o ruta local (p. ej. `C:\\Users\\<usuario>\\proyecto\\bank-full.json` o `file:///C:/ruta/archivo.csv`) que entregue CSV o JSON con columnas de features y `y`.
  - `AUTO_RETRAIN_AFTER_PREDICTION` (default: `true`) → reentrena en background después de cada predicción usando el dataset minado y ejemplos etiquetados.
  - `AUTO_RETRAIN` (default: `false`) y `RETRAIN_MIN_FEEDBACK` → reentrenamiento basado en ejemplos etiquetados vía `/feedback`.
  - `API_BASE_URL` (para el dashboard, si no usas localhost).
- Debes configurar `TRAINING_SQL` o `TRAINING_SOURCE_URL` para que el entrenamiento inicial funcione (no se descarga dataset).

## Correr Localmente

1) Crear entorno e instalar dependencias:

```
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2) (Opcional) Configurar Postgres local exportando `DATABASE_URL`, ejemplo:

```
set DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME
```

3) Levantar la API (crea tablas y entrena modelo inicial automáticamente):

```
uvicorn api.main:app --reload
```

- Salud: http://localhost:8000/health
- Predicción RL: POST http://localhost:8000/predict_rl
- Predicción DL: POST http://localhost:8000/predict_dl
- Métricas: GET http://localhost:8000/metrics

4) Levantar el dashboard (requiere `API_BASE_URL` si la API no está en localhost):

```
set API_BASE_URL=http://localhost:8000
streamlit run dashboard/app.py
```

## Despliegue Gratuito

### 1. Base de Datos: Neon (free Postgres)
- Crear una cuenta en https://neon.tech
- Crear proyecto y base de datos
- Obtener la cadena `postgresql://...`
- Convertir a formato SQLAlchemy: `postgresql+psycopg2://USER:PASSWORD@HOST/DB`
 - Añadir `?sslmode=require` si tu proveedor lo solicita (Neon lo requiere).

### 2. API en Render (Free Web Service)
- Conectar repo de GitHub a Render
- Crear un servicio Web:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
  - Añadir variable `DATABASE_URL` con la cadena de Neon
  - Alternativamente, Render detecta `render.yaml` en la raíz.

> Nota: Esta versión no descarga datasets. Debes proporcionar `TRAINING_SQL` o `TRAINING_SOURCE_URL` para que el entrenamiento inicial ocurra con tus datos.

### 3. Dashboard en Streamlit Community Cloud (gratis)
- Ir a https://share.streamlit.io/
- Conectar el repo
- Entry point: `dashboard/app.py`
- Configurar en Secrets: `API_BASE_URL: https://<tu-servicio-render>.onrender.com`

## Flujo del Sistema

1) Entrenamiento inicial (minado de datos): al iniciar la API, ejecuta la consulta SQL o la URL/ruta que definas y obtiene un dataset con columnas de entrada y la etiqueta `y`. El minado normaliza algunos campos (p. ej. sí/no, meses abreviados, categorías en español) y el pipeline de ML descarta `duration` (leakage) y normaliza `y`. Se entrenan dos pipelines: Regresión Logística (RL) y Deep Learning MLP (DL), ambos con `OneHotEncoder(handle_unknown="ignore")` + `StandardScaler`. Se almacenan métricas y versiones `vN`.

2) Predicción (dashboard → API): el usuario ingresa datos y elige RL o DL en el dashboard. La API recibe `POST /predict_rl` o `POST /predict_dl`, predice y devuelve probabilidad. Se guarda la predicción (RL) y se añade un ejemplo etiquetado (`y=pred`) para ambos. Si `AUTO_RETRAIN_AFTER_PREDICTION=true`, se reentrena en background el backend usado (RL o DL) y se crea una nueva versión con sus métricas.

3) Interacción: desde el dashboard puedes usar el modo Chat (texto libre) o el Formulario (campos en español con ayudas) para consultar una probabilidad de contratación clara (SÍ/NO + porcentaje). No se muestra “predicción 0/1”.

## Endpoints (resumen)

- `GET /health` → estado del servicio
- `GET /model/latest` → versión y métricas del último modelo
- `POST /predict_rl` → body `{ "features": { ... } }` → probabilidad, predicción, timestamp, versión (RL)
- `POST /predict_dl` → body `{ "features": { ... } }` → probabilidad, predicción, timestamp, versión (DL)
- `GET /metrics?limit=5` → historial de métricas por versión (incluye `model_type: rl|mlp`)
 - `POST /feedback` → guarda ejemplo etiquetado `{features, y}` para reentrenos manuales
 - `POST /retrain` → reentrena a partir del dataset base y CSV opcional (con `y`)

## Dashboard: Chat, Formulario y Métricas

- Accuracy, Precision, Recall, F1, ROC-AUC, (PR-AUC), Matriz de confusión
- Curva ROC, Curva Precision-Recall, Distribución de `y`
- Tendencia histórica por versión (líneas)
- Pestaña "Chat": escribe en lenguaje natural (p. ej., "Tengo 45 años, saldo 1200, hipoteca sí"). Un toggle "Usar Deep Learning" permite alternar entre RL y DL. El asistente responde en lenguaje claro: “Es probable que SÍ/NO contrates (≈ 78%)”. Incluye un panel "Ver más" con el endpoint usado (`/predict_rl` o `/predict_dl`), el JSON enviado y la respuesta.
- Pestaña "Formulario": campos en español con instrucciones (“Ingresa tu edad”, “¿Tienes hipoteca?”). Un toggle "Usar Deep Learning" alterna RL/DL. Incluye un panel "Ver más" con el endpoint, el JSON y la respuesta.
- Pestaña "Métricas": botón "Actualizar métricas" para refrescar y una tabla histórica con pestañas “Todos”, “RL”, “DL” que permite distinguir el backend (`metrics.model_type`).

## Notas

- La API por defecto permite CORS `*` para facilidad de demo.
- El pipeline maneja categorías desconocidas y realiza escalado en numéricos.
- El auto-reentrenamiento tras predicción usa la propia predicción como etiqueta (`y=pred`) para bootstrap de versiones (tanto RL como DL); en entornos reales se recomienda usar etiquetas humanas vía `/feedback` para evitar sesgos de auto-entrenamiento.

## Diferencias vs documentación previa (DataSet.pdf)

- Enfoque: la documentación anterior describe “minería de texto (PLN)”. El proyecto actual trabaja con datos tabulares del dataset Bank Marketing (UCI) y no realiza PLN.
- Visualización: antes se mencionaba un dashboard con Plotly/Dash. Ahora se utiliza Streamlit con pestañas de Chat, Formulario y Métricas.
- Preparación de datos: la doc previa usa `pd.get_dummies` y `MinMaxScaler` (0-1). El proyecto actual usa `OneHotEncoder(handle_unknown="ignore")` y `StandardScaler`.
- Persistencia de dataset procesado: la doc previa guardaba `bank-full-minado.csv`. En el proyecto actual el minado es dinámico desde `TRAINING_SQL` o `TRAINING_SOURCE_URL` (CSV/JSON/archivo local) y no se persiste un dataset procesado.
- Retraining: la doc previa no contemplaba reentrenamiento continuo. El proyecto actual puede reentrenar tras cada predicción (`AUTO_RETRAIN_AFTER_PREDICTION`) y también soporta feedback manual (`/feedback`).

## Licencia

Uso educativo y demostrativo.