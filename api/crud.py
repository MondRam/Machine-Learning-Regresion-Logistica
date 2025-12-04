from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from .models import ModelVersion, Prediction, LabeledExample


def get_latest_model(db: Session) -> Optional[ModelVersion]:
    return db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc())
    ).scalars().first()


def get_model_by_version(db: Session, version: str) -> Optional[ModelVersion]:
    return db.execute(
        select(ModelVersion).where(ModelVersion.version == version)
    ).scalars().first()


def create_model_version(db: Session, version: str, artifact: bytes, metrics: Dict[str, Any]) -> ModelVersion:
    mv = ModelVersion(version=version, artifact=artifact, metrics=metrics)
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


def save_prediction(
    db: Session,
    *,
    features: Dict[str, Any],
    predicted: int,
    probability: float,
    model: ModelVersion,
) -> Prediction:
    p = Prediction(features=features, predicted=predicted, probability=probability, model=model)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def get_metrics_history(db: Session, limit: int = 10) -> List[ModelVersion]:
    return db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
    ).scalars().all()


def next_version(db: Session) -> str:
    latest = db.execute(select(func.max(ModelVersion.id))).scalar()
    if latest is None:
        return "v1"
    return f"v{int(latest) + 1}"


# --- Labeled feedback ---
def add_labeled_example(db: Session, *, features: Dict[str, Any], y: int) -> LabeledExample:
    row = LabeledExample(features=features, y=int(y))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_all_labeled_examples(db: Session) -> List[LabeledExample]:
    return db.execute(select(LabeledExample).order_by(LabeledExample.created_at.asc())).scalars().all()


def count_labeled_examples(db: Session) -> int:
    return int(db.execute(select(func.count()).select_from(LabeledExample)).scalar() or 0)