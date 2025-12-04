from __future__ import annotations

import os
import csv
from io import StringIO
from typing import Tuple
from urllib.parse import urlparse, unquote

import pandas as pd
import requests

from .db import engine


# Mapas de normalización Español → valores esperados por el modelo/API
YES_SET = {"si", "sí", "yes", "y", "true", "1"}
NO_SET = {"no", "false", "0"}

JOB_ES_TO_EN = {
    "Administración": "admin.",
    "Obrero": "blue-collar",
    "Emprendedor": "entrepreneur",
    "Empleada doméstica": "housemaid",
    "Gerencia": "management",
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
    "Básica 4 años": "basic.4y",
    "Básica 6 años": "basic.6y",
    "Básica 9 años": "basic.9y",
    "Secundaria": "high.school",
    "Analfabeta": "illiterate",
    "Curso profesional": "professional.course",
    "Universidad": "university.degree",
    "Desconocido": "unknown",
}

CONTACT_ES_TO_EN = {"Celular": "cellular", "Teléfono fijo": "telephone"}

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
    "lun": "mon",
    "mar": "tue",
    "mié": "wed",
    "mie": "wed",
    "jue": "thu",
    "vie": "fri",
}


def _maybe_lower(x):
    return str(x).strip().lower() if pd.notna(x) else x


def _map_yes_no_unk(x):
    t = _maybe_lower(x)
    if t in YES_SET:
        return "yes"
    if t in NO_SET:
        return "no"
    return "unknown"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Renombrar columnas comunes en español → columnas del dataset/modelo
    rename_map = {
        "edad": "age",
        "ocupacion": "job",
        "trabajo": "job",
        "estado_civil": "marital",
        "educacion": "education",
        "mora": "default",
        "en_mora": "default",
        "saldo": "balance",
        "hipoteca": "housing",
        "prestamo": "loan",
        "contacto": "contact",
        "dia": "day",
        "dia_mes": "day",
        "mes": "month",
        "dia_semana": "day_of_week",
        "campania": "campaign",
        "campaña": "campaign",
        "dias_desde_contacto": "pdays",
        "pdays": "pdays",
        "previos": "previous",
        "contactos_previos": "previous",
        "resultado_previo": "poutcome",
        "tasa_empleo": "emp.var.rate",
        "emp_var_rate": "emp.var.rate",
        "indice_precios": "cons.price.idx",
        "cons_price_idx": "cons.price.idx",
        "indice_confianza": "cons.conf.idx",
        "cons_conf_idx": "cons.conf.idx",
        "euribor": "euribor3m",
        "empleados": "nr.employed",
        "objetivo": "y",
        "contrata": "y",
    }

    cols = {c: rename_map.get(c.lower(), c) for c in df.columns}
    df = df.rename(columns=cols)

    # Normalizar categorías en español
    if "job" in df.columns:
        df["job"] = df["job"].map(lambda v: JOB_ES_TO_EN.get(str(v), str(v)))
    if "marital" in df.columns:
        df["marital"] = df["marital"].map(lambda v: MARITAL_ES_TO_EN.get(str(v), str(v)))
    if "education" in df.columns:
        df["education"] = df["education"].map(lambda v: EDU_ES_TO_EN.get(str(v), str(v)))
    if "default" in df.columns:
        df["default"] = df["default"].map(_map_yes_no_unk)
    if "housing" in df.columns:
        df["housing"] = df["housing"].map(_map_yes_no_unk)
    if "loan" in df.columns:
        df["loan"] = df["loan"].map(_map_yes_no_unk)
    if "contact" in df.columns:
        df["contact"] = df["contact"].map(lambda v: CONTACT_ES_TO_EN.get(str(v), str(v)))
    if "month" in df.columns:
        df["month"] = df["month"].map(lambda v: MONTH_ES_TO_EN.get(_maybe_lower(v), str(v)))
    if "day_of_week" in df.columns:
        df["day_of_week"] = df["day_of_week"].map(lambda v: WDAY_ES_TO_EN.get(_maybe_lower(v), str(v)))

    # Coerción de tipos numéricos cuando aplican
    numeric_like = [
        "age",
        "balance",
        "day",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    for c in numeric_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalizar target si viene en texto español
    if "y" in df.columns:
        y_norm = df["y"].astype(str).str.lower().str.strip()
        df["y"] = y_norm.map({"yes": 1, "no": 0, "1": 1, "0": 0, "si": 1, "sí": 1}).fillna(y_norm).astype(str)
    return df


def collect_training_data() -> pd.DataFrame:
    """Obtiene datos de entrenamiento desde una fuente externa (sin dataset embebido).

    Opciones de configuración (variables de entorno):
    - TRAINING_SQL: consulta SQL a ejecutar sobre DATABASE_URL. Debe devolver columnas de features y "y".
    - TRAINING_SOURCE_URL: URL que devuelve CSV o JSON con las columnas requeridas.
    """

    sql = os.getenv("TRAINING_SQL")
    source_url = os.getenv("TRAINING_SOURCE_URL")

    df: pd.DataFrame | None = None

    if sql:
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn)
    elif source_url:
        # Permitir tanto URLs HTTP(S) como rutas locales (incluyendo file://)
        is_http = source_url.lower().startswith(("http://", "https://"))

        if is_http:
            r = requests.get(source_url, timeout=60)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            text = r.text
            if "application/json" in ctype or text.strip().startswith("[") or text.strip().startswith("{"):
                # JSON: array de objetos o un objeto con clave "data"
                try:
                    payload = r.json()
                except Exception:
                    payload = None
                if isinstance(payload, list):
                    df = pd.DataFrame(payload)
                elif isinstance(payload, dict):
                    if "data" in payload and isinstance(payload["data"], list):
                        df = pd.DataFrame(payload["data"])
                    else:
                        # Último recurso: DataFrame de un único objeto
                        df = pd.DataFrame([payload])
                else:
                    # Fallback si no pudimos decodificar JSON
                    # Intentar CSV auto-delimited (coma/semicolon)
                    try:
                        sample = text[:2048]
                        try:
                            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                            sep = dialect.delimiter
                        except Exception:
                            sep = ","
                        df = pd.read_csv(StringIO(text), sep=sep)
                    except Exception as e:
                        raise RuntimeError(f"No se pudo interpretar la respuesta HTTP como JSON/CSV: {e}")
            else:
                # CSV: autodetectar delimitador , o ;
                sample = text[:2048]
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                    sep = dialect.delimiter
                except Exception:
                    sep = ","
                df = pd.read_csv(StringIO(text), sep=sep)
        else:
            # Ruta local (Windows o Unix) o esquema file://
            path = source_url
            if source_url.lower().startswith("file://"):
                parsed = urlparse(source_url)
                path = unquote(parsed.path)
                # En Windows urlparse('file:///C:/...').path -> '/C:/...'
                if os.name == "nt" and path.startswith("/") and len(path) > 3 and path[2] == ":":
                    path = path[1:]

            if not os.path.exists(path):
                raise RuntimeError(f"No existe el archivo local: {path}")

            _, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext == ".json":
                # Leer JSON de lista de objetos o dict con 'data'
                import json

                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    try:
                        payload = json.loads(content)
                    except Exception as e:
                        raise RuntimeError(f"JSON inválido en {path}: {e}")

                if isinstance(payload, list):
                    df = pd.DataFrame(payload)
                elif isinstance(payload, dict):
                    if "data" in payload and isinstance(payload["data"], list):
                        df = pd.DataFrame(payload["data"])
                    else:
                        df = pd.DataFrame([payload])
                else:
                    raise RuntimeError("Formato JSON no soportado: se espera lista de objetos o dict con 'data'.")
            else:
                # CSV local con autodetección de delimitador
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        sample = f.read(2048)
                        try:
                            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                            sep = dialect.delimiter
                        except Exception:
                            sep = ","
                except FileNotFoundError:
                    raise RuntimeError(f"No se pudo abrir el archivo: {path}")

                df = pd.read_csv(path, sep=sep)
    else:
        raise RuntimeError(
            "No se configuró TRAINING_SQL ni TRAINING_SOURCE_URL para minado de datos."
        )

    if df is None or df.empty:
        raise RuntimeError("La fuente de datos para entrenamiento no devolvió registros.")

    df = _normalize_columns(df)

    # Asegurar presencia de 'y'
    if "y" not in df.columns:
        raise RuntimeError("El dataset minado no contiene la columna objetivo 'y'.")

    # El pipeline interno descarta 'duration' y normaliza 'y', por lo que devolvemos tal cual
    return df