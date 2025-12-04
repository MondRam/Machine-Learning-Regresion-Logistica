from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .db import Base, engine, get_db, SessionLocal
from . import crud
from .ml import (
    load_default_dataset,
    train_and_evaluate,
    dump_artifact,
    load_artifact,
)
from .data_mining import collect_training_data
from .schemas import (
    PredictRequest,
    PredictResponse,
    PredictBothResponse,
    PredictOption,
    MetricsHistoryResponse,
    RetrainResponse,
    ModelInfo,
    FeedbackRequest,
    FeedbackResponse,
)


app = FastAPI(title="Bank Marketing - Logistic Regression API")

PIPE_LOGREG = None
PIPE_MLP = None
SCHEMA_FOR_BOTH = None
ALL_CAT_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
ALL_NUM_COLS = [
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Create tables
    Base.metadata.create_all(bind=engine)
    # Ensure at least one model exists
    db = SessionLocal()
    try:
        # Intentar cargar modelos existentes desde la base (evita entrenar pesado en startup)
        try:
            hist = crud.get_metrics_history(db, limit=20)
            for r in hist:
                m = r.metrics or {}
                t = str(m.get("model_type", "")).lower()
                if t in {"mlp", "torch", "pytorch"} and globals()["PIPE_MLP"] is None:
                    try:
                        globals()["PIPE_MLP"] = load_artifact(r.artifact)
                        globals()["SCHEMA_FOR_BOTH"] = globals()["SCHEMA_FOR_BOTH"] or m.get("schema")
                        break
                    except Exception:
                        pass
            for r in hist:
                m = r.metrics or {}
                t = str(m.get("model_type", "")).lower()
                if t in {"logreg", "lr", "classic", "sklearn"} and globals()["PIPE_LOGREG"] is None:
                    try:
                        globals()["PIPE_LOGREG"] = load_artifact(r.artifact)
                        globals()["SCHEMA_FOR_BOTH"] = globals()["SCHEMA_FOR_BOTH"] or m.get("schema")
                        break
                    except Exception:
                        pass
        except Exception:
            pass
        latest = crud.get_latest_model(db)
        if latest is None:
            try:
                base_df = collect_training_data()
            except Exception:
                base_df = load_default_dataset()
            try:
                _ss = int(os.getenv("STARTUP_SAMPLE_SIZE", "2000"))
            except Exception:
                _ss = 2000
            if _ss > 0 and len(base_df) > _ss:
                base_df = base_df.sample(n=_ss, random_state=42)
            pipe, metrics, _schema = train_and_evaluate(base_df)
            try:
                m = dict(metrics)
                m["model_type"] = "logreg"
                metrics = m
            except Exception:
                pass
            version = crud.next_version(db)
            artifact = dump_artifact(pipe)
            crud.create_model_version(db, version=version, artifact=artifact, metrics=metrics)

        try:
            enable_mlp = os.getenv("ENABLE_MLP_STARTUP", "false").lower() in {"1", "true", "yes", "y", "on"}
            enable_mlp_bg = os.getenv("ENABLE_MLP_STARTUP_BACKGROUND", "false").lower() in {"1", "true", "yes", "y", "on"}
            try:
                df = collect_training_data()
            except Exception:
                df = load_default_dataset()
            try:
                sample_s = int(os.getenv("STARTUP_SAMPLE_SIZE", "2000"))
            except Exception:
                sample_s = 2000
            if sample_s > 0 and len(df) > sample_s:
                df = df.sample(n=sample_s, random_state=42)
            if globals()["PIPE_LOGREG"] is None:
                pipe_lr, metrics_lr, schema_lr = train_and_evaluate(df, model_type="logreg")
                globals()["PIPE_LOGREG"] = pipe_lr
                globals()["SCHEMA_FOR_BOTH"] = schema_lr
            if enable_mlp:
                try:
                    if globals()["PIPE_MLP"] is None:
                        pipe_mlp, metrics_mlp, schema_mlp = train_and_evaluate(df, model_type="mlp")
                        globals()["PIPE_MLP"] = pipe_mlp
                        globals()["SCHEMA_FOR_BOTH"] = globals()["SCHEMA_FOR_BOTH"] or schema_mlp
                except Exception:
                    pass
            elif enable_mlp_bg:
                import threading
                def _bg():
                    try:
                        if globals()["PIPE_MLP"] is None:
                            pipe_mlp, metrics_mlp, schema_mlp = train_and_evaluate(df, model_type="mlp")
                            globals()["PIPE_MLP"] = pipe_mlp
                    except Exception:
                        pass
                threading.Thread(target=_bg, daemon=True).start()
            # Cold‑start: si aún no hay MLP, inicializar uno ligero para evitar 503
            if globals()["PIPE_MLP"] is None:
                try:
                    cs = int(os.getenv("MLP_COLD_START_SAMPLE", "500"))
                except Exception:
                    cs = 500
                df_cs = df.sample(n=min(cs, len(df)), random_state=42) if len(df) > cs else df
                prev_svd = os.getenv("MODEL_MLP_SVD_COMPONENTS")
                prev_ep = os.getenv("MODEL_MLP_MAX_EPOCHS")
                prev_hid = os.getenv("MODEL_MLP_HIDDEN")
                os.environ["MODEL_MLP_SVD_COMPONENTS"] = "32"
                os.environ["MODEL_MLP_MAX_EPOCHS"] = "1"
                os.environ["MODEL_MLP_HIDDEN"] = "32,16"
                try:
                    pipe_mlp_cs, metrics_mlp_cs, schema_mlp_cs = train_and_evaluate(df_cs, model_type="mlp")
                    globals()["PIPE_MLP"] = pipe_mlp_cs
                    try:
                        mcs = dict(metrics_mlp_cs)
                        mcs["model_type"] = "mlp"
                        vcs = crud.next_version(db)
                        acs = dump_artifact(pipe_mlp_cs)
                        crud.create_model_version(db, version=vcs, artifact=acs, metrics=mcs)
                        globals()["SCHEMA_FOR_BOTH"] = globals()["SCHEMA_FOR_BOTH"] or schema_mlp_cs
                    except Exception:
                        pass
                except Exception:
                    pass
                finally:
                    if prev_svd is not None:
                        os.environ["MODEL_MLP_SVD_COMPONENTS"] = prev_svd
                    else:
                        os.environ.pop("MODEL_MLP_SVD_COMPONENTS", None)
                    if prev_ep is not None:
                        os.environ["MODEL_MLP_MAX_EPOCHS"] = prev_ep
                    else:
                        os.environ.pop("MODEL_MLP_MAX_EPOCHS", None)
                    if prev_hid is not None:
                        os.environ["MODEL_MLP_HIDDEN"] = prev_hid
                    else:
                        os.environ.pop("MODEL_MLP_HIDDEN", None)
        except Exception:
            pass
    finally:
        db.close()


# Auto-retrain settings (disabled by default for a simpler experience)
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() in {"1", "true", "yes", "y", "on"}
# Nuevo: reentrenar después de cada predicción (controlado por variable de entorno)
AUTO_RETRAIN_AFTER_PREDICTION = os.getenv("AUTO_RETRAIN_AFTER_PREDICTION", "true").lower() in {"1", "true", "yes", "y", "on"}
try:
    RETRAIN_MIN_FEEDBACK = int(os.getenv("RETRAIN_MIN_FEEDBACK", "1"))
except Exception:
    RETRAIN_MIN_FEEDBACK = 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/latest", response_model=ModelInfo)
def model_latest(db: Session = Depends(get_db)):
    mv = crud.get_latest_model(db)
    if mv is None:
        raise HTTPException(status_code=500, detail="Model not available")
    return {
        "version": mv.version,
        "created_at": mv.created_at,
        "metrics": mv.metrics,
    }


@app.post("/predict_rl", response_model=PredictResponse)
def predict_rl(req: PredictRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    pipe = PIPE_LOGREG
    if pipe is None:
        raise HTTPException(status_code=500, detail="Classic model not initialized")

    # Derivar columnas esperadas y construir fila exacta
    cat_cols: list[str] = []
    num_cols: list[str] = []
    required: list[str] = []
    X_in = None
    if hasattr(pipe, "named_steps"):
        pre = pipe.named_steps.get("preprocess")
        if pre is not None and hasattr(pre, "transformers_"):
            for name, trans, cols in pre.transformers_:
                cols = list(cols)
                if name == "cat":
                    cat_cols = cols
                elif name == "num":
                    num_cols = cols
                required.extend(cols)
    if not required:
        if SCHEMA_FOR_BOTH:
            cat_cols = list(SCHEMA_FOR_BOTH.get("categorical", []))
            num_cols = list(SCHEMA_FOR_BOTH.get("numerical", []))
            required = list(SCHEMA_FOR_BOTH.get("all", cat_cols + num_cols))
        else:
            cat_cols = list(ALL_CAT_COLS)
            num_cols = list(ALL_NUM_COLS)
            required = list(ALL_CAT_COLS + ALL_NUM_COLS)
    row = {}
    for c in required:
        if c in cat_cols:
            row[c] = req.features.get(c, "unknown")
        else:
            row[c] = req.features.get(c, 0)
    X_in = pd.DataFrame([row], columns=required)
    try:
        print("[debug RL] required_cat", len(cat_cols), cat_cols[:5])
        print("[debug RL] required_num", len(num_cols), num_cols[:5])
        print("[debug RL] x_cols", len(list(X_in.columns)), list(X_in.columns)[:5])
    except Exception:
        pass

    # Predict
    try:
        _proba = pipe.predict_proba(X_in)
        if hasattr(_proba, "shape") and len(getattr(_proba, "shape", ())) == 2 and _proba.shape[1] >= 2:
            proba = float(_proba[0, 1])
        else:
            proba = float(np.array(_proba).reshape(-1)[0])
        pred = int(1 if proba >= 0.5 else 0)
    except Exception as e:
        msg = str(e)
        try:
            import re
            m = re.search(r"missing:\s*\{([^}]*)\}", msg)
            if m:
                missing_raw = m.group(1)
                missing = [s.strip().strip("'\"") for s in missing_raw.split(',') if s.strip()]
                for col in missing:
                    if col in cat_cols:
                        X_in[col] = "unknown"
                    else:
                        X_in[col] = 0
                all_cols2 = (cat_cols + num_cols) if (cat_cols or num_cols) else list(X_in.columns)
                X_in = X_in.reindex(columns=all_cols2, fill_value=0)
                _proba = pipe.predict_proba(X_in)
                if hasattr(_proba, "shape") and len(getattr(_proba, "shape", ())) == 2 and _proba.shape[1] >= 2:
                    proba = float(_proba[0, 1])
                else:
                    proba = float(np.array(_proba).reshape(-1)[0])
                pred = int(1 if proba >= 0.5 else 0)
            else:
                raise
        except Exception:
            try:
                dbg = {
                    "cat_cols": cat_cols,
                    "num_cols": num_cols,
                    "x_cols": list(X_in.columns),
                }
                raise HTTPException(status_code=400, detail=f"Prediction error: {e}; dbg={dbg}")
            except Exception:
                raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    p = None
    try:
        mv = crud.get_latest_model(db)
        if mv is not None:
            p = crud.save_prediction(
                db,
                features=req.features,
                predicted=pred,
                probability=proba,
                model=mv,
            )
    except Exception:
        p = None

    # Guardar ejemplo etiquetado automáticamente con el resultado de la predicción
    try:
        crud.add_labeled_example(db, features=req.features, y=int(pred))
        print("[auto-retrain] Ejemplo etiquetado agregado desde predicción")
    except Exception as e:
        print(f"[auto-retrain] No se pudo agregar ejemplo etiquetado: {e}")

    # Disparar reentrenamiento en background si está habilitado
    if AUTO_RETRAIN_AFTER_PREDICTION:
        try:
            background_tasks.add_task(_retrain_from_feedback_background, "logreg")
        except Exception:
            pass

    return PredictResponse(
        predicted=pred,
        probability=proba,
        timestamp=(p.created_at if p else datetime.utcnow()),
        model_version="logreg",
    )


def _labeled_examples_to_df(examples: list[Dict[str, Any]]) -> pd.DataFrame:
    if not examples:
        return pd.DataFrame()
    rows = []
    for ex in examples:
        # Expect keys: features (dict), y (int)
        row = dict(ex["features"])  # shallow copy
        row["y"] = int(ex["y"])
        rows.append(row)
    return pd.DataFrame(rows)


def _retrain_from_feedback_background(model_type: str = "logreg"):
    print("[auto-retrain] Inicio de reentrenado en background")
    db = SessionLocal()
    try:
        try:
            base_df = collect_training_data()
        except Exception as e:
            print(f"[auto-retrain] Error al minar datos de entrenamiento: {e}")
            try:
                base_df = load_default_dataset()
                print("[auto-retrain] Usando dataset por defecto para reentrenar")
            except Exception:
                raise
        labeled = crud.get_all_labeled_examples(db)
        try:
            max_fb = int(os.getenv("RETRAIN_FEEDBACK_MAX", "2000"))
        except Exception:
            max_fb = 2000
        if max_fb > 0 and len(labeled) > max_fb:
            labeled = labeled[-max_fb:]
        labeled_dicts = [{"features": r.features, "y": r.y} for r in labeled]
        add_df = _labeled_examples_to_df(labeled_dicts)

        mt = (model_type or "logreg").lower()
        prev_svd = os.getenv("MODEL_MLP_SVD_COMPONENTS")
        prev_ep = os.getenv("MODEL_MLP_MAX_EPOCHS")
        prev_hid = os.getenv("MODEL_MLP_HIDDEN")
        try:
            if mt in {"mlp", "torch", "pytorch"}:
                try:
                    rs = int(os.getenv("RETRAIN_SAMPLE_SIZE", "1500"))
                except Exception:
                    rs = 1500
                if rs > 0 and len(base_df) > rs:
                    base_df = base_df.sample(n=rs, random_state=42)
                os.environ["MODEL_MLP_SVD_COMPONENTS"] = os.getenv("RETRAIN_MLP_SVD_COMPONENTS", os.getenv("MODEL_MLP_SVD_COMPONENTS", "32"))
                os.environ["MODEL_MLP_MAX_EPOCHS"] = os.getenv("RETRAIN_MLP_MAX_EPOCHS", os.getenv("MODEL_MLP_MAX_EPOCHS", "2"))
                os.environ["MODEL_MLP_HIDDEN"] = os.getenv("RETRAIN_MLP_HIDDEN", os.getenv("MODEL_MLP_HIDDEN", "32,16"))

            pipe, metrics, _schema = train_and_evaluate(base_df, additional_df=add_df if not add_df.empty else None, model_type=model_type)
        finally:
            if prev_svd is not None:
                os.environ["MODEL_MLP_SVD_COMPONENTS"] = prev_svd
            else:
                os.environ.pop("MODEL_MLP_SVD_COMPONENTS", None)
            if prev_ep is not None:
                os.environ["MODEL_MLP_MAX_EPOCHS"] = prev_ep
            else:
                os.environ.pop("MODEL_MLP_MAX_EPOCHS", None)
            if prev_hid is not None:
                os.environ["MODEL_MLP_HIDDEN"] = prev_hid
            else:
                os.environ.pop("MODEL_MLP_HIDDEN", None)

        metrics = dict(metrics)
        metrics["model_type"] = model_type
        version = crud.next_version(db)
        artifact = dump_artifact(pipe)
        mv = crud.create_model_version(db, version=version, artifact=artifact, metrics=metrics)

        if (model_type or "logreg").lower() in {"mlp", "torch", "pytorch"}:
            globals()["PIPE_MLP"] = pipe
        else:
            globals()["PIPE_LOGREG"] = pipe

        print(f"[auto-retrain] Nueva versión creada: {mv.version}")
    finally:
        print("[auto-retrain] Fin de reentrenado en background")
        db.close()


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Guardar ejemplo etiquetado
    crud.add_labeled_example(db, features=payload.features, y=int(payload.y))

    retrain_started = False
    if AUTO_RETRAIN:
        try:
            total = crud.count_labeled_examples(db)
            if RETRAIN_MIN_FEEDBACK <= 1 or (total % max(1, RETRAIN_MIN_FEEDBACK) == 0):
                background_tasks.add_task(_retrain_from_feedback_background)
                retrain_started = True
        except Exception:
            retrain_started = False

    return FeedbackResponse(accepted=True, retrain_started=retrain_started)


@app.get("/metrics", response_model=MetricsHistoryResponse)
def metrics(limit: int = 5, db: Session = Depends(get_db)):
    rows = crud.get_metrics_history(db, limit=limit)
    history = [
        {
            "version": r.version,
            "created_at": r.created_at,
            "metrics": r.metrics,
        }
        for r in rows
    ]
    return {"history": history}


def _read_csv_auto(content: bytes) -> pd.DataFrame:
    # Detectar separador automáticamente (',' o ';')
    import csv
    sample = content[:2048].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        sep = dialect.delimiter
    except Exception:
        sep = ","
    return pd.read_csv(io.BytesIO(content), sep=sep)


@app.post("/retrain", response_model=RetrainResponse)
def retrain(
    labeled_csv: Optional[UploadFile] = File(default=None, description="Optional CSV with same schema including 'y' target"),
    db: Session = Depends(get_db),
):
    base_df = load_default_dataset()
    add_df = None
    if labeled_csv is not None:
        try:
            content = labeled_csv.file.read()
            add_df = _read_csv_auto(content)
            if "y" not in add_df.columns:
                raise ValueError("Provided CSV must include 'y' target")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    pipe, metrics, _schema = train_and_evaluate(base_df, additional_df=add_df)
    version = crud.next_version(db)
    artifact = dump_artifact(pipe)
    mv = crud.create_model_version(db, version=version, artifact=artifact, metrics=metrics)

    return RetrainResponse(version=mv.version, created_at=mv.created_at, metrics=mv.metrics)
@app.post("/predict_dl", response_model=PredictResponse)
def predict_dl(req: PredictRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if PIPE_MLP is None:
        raise HTTPException(status_code=503, detail="Deep model warming up; retry later")

    cat_cols: list[str] = []
    num_cols: list[str] = []
    required: list[str] = []
    pre = PIPE_MLP.named_steps.get("preprocess") if hasattr(PIPE_MLP, "named_steps") else None
    if pre is not None and hasattr(pre, "transformers_"):
        for name, trans, cols in pre.transformers_:
            cols = list(cols)
            if name == "cat":
                cat_cols = cols
            elif name == "num":
                num_cols = cols
            required.extend(cols)
    if not required:
        if SCHEMA_FOR_BOTH:
            cat_cols = list(SCHEMA_FOR_BOTH.get("categorical", []))
            num_cols = list(SCHEMA_FOR_BOTH.get("numerical", []))
            required = list(SCHEMA_FOR_BOTH.get("all", cat_cols + num_cols))
        else:
            cat_cols = list(ALL_CAT_COLS)
            num_cols = list(ALL_NUM_COLS)
            required = list(ALL_CAT_COLS + ALL_NUM_COLS)
    row = {}
    for c in required:
        if c in cat_cols:
            row[c] = req.features.get(c, "unknown")
        else:
            row[c] = req.features.get(c, 0)
    X_in = pd.DataFrame([row], columns=required)

    try:
        proba_dl_raw = PIPE_MLP.predict_proba(X_in)
        if hasattr(proba_dl_raw, "shape") and len(getattr(proba_dl_raw, "shape", ())) == 2 and proba_dl_raw.shape[1] >= 2:
            proba = float(proba_dl_raw[0, 1])
        else:
            proba = float(np.array(proba_dl_raw).reshape(-1)[0])
        pred = int(1 if proba >= 0.5 else 0)
    except Exception as e:
        msg = str(e)
        try:
            import re
            m = re.search(r"missing:\s*\{([^}]*)\}", msg)
            if m:
                missing_raw = m.group(1)
                missing = [s.strip().strip("'\"") for s in missing_raw.split(',') if s.strip()]
                for col in missing:
                    if col in cat_cols:
                        X_in[col] = "unknown"
                    else:
                        X_in[col] = 0
                all_cols2 = (cat_cols + num_cols) if (cat_cols or num_cols) else list(X_in.columns)
                X_in = X_in.reindex(columns=all_cols2, fill_value=0)
                proba_dl_raw = PIPE_MLP.predict_proba(X_in)
                if hasattr(proba_dl_raw, "shape") and len(getattr(proba_dl_raw, "shape", ())) == 2 and proba_dl_raw.shape[1] >= 2:
                    proba = float(proba_dl_raw[0, 1])
                else:
                    proba = float(np.array(proba_dl_raw).reshape(-1)[0])
                pred = int(1 if proba >= 0.5 else 0)
            else:
                raise
        except Exception:
            try:
                dbg = {
                    "cat_cols": cat_cols,
                    "num_cols": num_cols,
                    "x_cols": list(X_in.columns),
                }
                raise HTTPException(status_code=400, detail=f"Prediction error: {e}; dbg={dbg}")
            except Exception:
                raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    try:
        crud.add_labeled_example(db, features=req.features, y=int(pred))
    except Exception:
        pass

    if AUTO_RETRAIN_AFTER_PREDICTION:
        try:
            background_tasks.add_task(_retrain_from_feedback_background, "mlp")
        except Exception:
            pass

    return PredictResponse(
        predicted=pred,
        probability=proba,
        timestamp=datetime.utcnow(),
        model_version="mlp",
    )
