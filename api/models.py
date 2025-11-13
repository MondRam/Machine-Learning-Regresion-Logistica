from sqlalchemy import Table, Column, Integer, String, Float, DateTime, LargeBinary, MetaData, JSON
from .config import DB

# Metadata para agrupar las definiciones
metadata = MetaData()


# -----------------------------
# Tabla de datos de entrenamiento
# -----------------------------
insertar_datos = Table(
    "insertar_datos", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("age", Integer),
    Column("job", String),
    Column("marital", String),
    Column("education", String),
    Column("balance", Float),
    Column("housing", String),
    Column("loan", String),
    Column("y", Integer)
)

# -----------------------------
# Tabla de métricas del modelo
# -----------------------------
metricas = Table(
    "metricas", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime),
    Column("modelo", String),
    Column("accuracy", Float),
    Column("precision", Float),
    Column("recall", Float),
    Column("f1", Float),
    Column("matriz_confusion", JSON),
    Column("pr_precision", JSON),
    Column("pr_recall", JSON)
)


# -----------------------------
# Tabla de modelos persistidos
# -----------------------------
modelos = Table(
    "modelos", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime),
    Column("modelo", LargeBinary)   # Aquí se guarda el .pkl en formato binario (BYTEA en Postgres)
)

# -----------------------------
# Crear todas las tablas si no existen
# -----------------------------
def init_db():
    metadata.create_all(DB)
