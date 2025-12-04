from __future__ import annotations

import io
import os
import joblib
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

try:
    import torch
    import torch.nn as nn

    class TorchMLP(nn.Module):
        def __init__(self, hidden_dims=(128, 64), dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.LazyLinear(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[1], 1),
            )

        def forward(self, x):
            return self.net(x.float())
except Exception:
    TorchMLP = None


UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
CSV_PATH_IN_ZIP = "bank-additional/bank-additional-full.csv"


def _download_bank_dataset() -> pd.DataFrame:
    r = requests.get(UCI_ZIP_URL, timeout=60)
    r.raise_for_status()
    import zipfile

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    with zf.open(CSV_PATH_IN_ZIP) as f:
        df = pd.read_csv(f, sep=";")
    return df


def _load_local_dataset_if_available() -> pd.DataFrame | None:
    # Permite override explícito por variable de entorno
    override = os.getenv("BANK_DATASET_PATH")
    candidates: list[str] = []
    if override:
        candidates.append(override)
    # Buscar en la raíz del proyecto
    base_dir = os.path.dirname(os.path.dirname(__file__))
    candidates.extend([
        os.path.join(base_dir, "bank-additional-full.csv"),
        os.path.join(base_dir, "bank-full.csv"),
    ])

    for path in candidates:
        if path and os.path.isfile(path):
            # Intentar detectar separador (coma o punto y coma)
            try:
                # Probar ';' primero (común en este dataset)
                df = pd.read_csv(path, sep=";")
            except Exception:
                df = pd.read_csv(path)
            return df
    return None


def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Drop duration (leakage as per dataset docs)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Target y can come as {yes/no}, {"1"/"0"} or {1/0}
    y_raw = df["y"]
    if y_raw.dtype == object:
        y_norm = y_raw.astype(str).str.lower().str.strip()
        y = y_norm.map({"yes": 1, "no": 0, "1": 1, "0": 0}).astype(int)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=["y"]).copy()
    return X, y


def _build_pipeline(X: pd.DataFrame, model_type: str | None = None) -> Tuple[Pipeline, Dict[str, Any]]:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    model_type = (model_type or os.getenv("MODEL_TYPE", "logreg")).lower()

    use_mlp = model_type in {"mlp", "torch", "pytorch"}

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    transformers = []
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", onehot),
                    ]
                ),
                cat_cols,
            )
        )
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ]
                ),
                num_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    clf: Any
    if use_mlp and TorchMLP is not None:
        from skorch import NeuralNetBinaryClassifier
        hidden_str = os.getenv("MODEL_MLP_HIDDEN", "128,64")
        try:
            hidden_dims = tuple(int(h) for h in hidden_str.split(",") if h)
        except Exception:
            hidden_dims = (128, 64)
        dropout = float(os.getenv("MODEL_MLP_DROPOUT", "0.1"))
        max_epochs = int(os.getenv("MODEL_MLP_MAX_EPOCHS", "20"))
        lr = float(os.getenv("MODEL_MLP_LR", "0.001"))

        clf = NeuralNetBinaryClassifier(
            module=TorchMLP,
            module__hidden_dims=hidden_dims,
            module__dropout=dropout,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.Adam,
            lr=lr,
            max_epochs=max_epochs,
            train_split=None,
            iterator_train__shuffle=True,
            verbose=0,
            device="cpu",
        )
    else:
        clf = LogisticRegression(max_iter=200, class_weight="balanced")

    steps = [("preprocess", preprocessor)]
    if use_mlp:
        try:
            svd_components = int(os.getenv("MODEL_MLP_SVD_COMPONENTS", "128"))
        except Exception:
            svd_components = 128
        steps.append(("svd", TruncatedSVD(n_components=svd_components, random_state=42)))
    steps.append(("clf", clf))
    pipe = Pipeline(steps=steps)

    schema = {
        "categorical": cat_cols,
        "numerical": num_cols,
        "all": cat_cols + num_cols,
    }
    return pipe, schema


def train_and_evaluate(base_df: pd.DataFrame, additional_df: pd.DataFrame | None = None, model_type: str | None = None) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    if additional_df is not None and not additional_df.empty:
        # Expect same columns and target y
        base_df = pd.concat([base_df, additional_df], axis=0, ignore_index=True)

    X, y = _prepare_dataframe(base_df)
    pipe, schema = _build_pipeline(X, model_type=model_type)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if (model_type or os.getenv("MODEL_TYPE", "logreg")).lower() in {"mlp", "torch", "pytorch"}:
        y_train = y_train.astype(np.float32)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    _proba = pipe.predict_proba(X_test)
    if hasattr(_proba, "shape") and len(getattr(_proba, "shape", ())) == 2 and _proba.shape[1] >= 2:
        y_proba = _proba[:, 1]
    else:
        y_proba = np.array(_proba).reshape(-1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rocauc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)

    y_dist = {
        "0": int((y == 0).sum()),
        "1": int((y == 1).sum()),
    }

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(rocauc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "y_distribution": y_dist,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": prec_c.tolist(), "recall": rec_c.tolist()},
        "schema": schema,
    }

    return pipe, metrics, schema


def load_default_dataset() -> pd.DataFrame:
    # Preferir el CSV local si el usuario lo colocó en el proyecto
    df = _load_local_dataset_if_available()
    if df is not None:
        return df
    # Fallback: descargar de UCI
    return _download_bank_dataset()


def dump_artifact(pipe: Pipeline) -> bytes:
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    return buf.getvalue()


def load_artifact(data: bytes) -> Pipeline:
    buf = io.BytesIO(data)
    pipe = joblib.load(buf)
    return pipe
