"""
src/serving/app.py

FastAPI inference endpoint for ARS-Stack — the "6. Evaluation & Serving"
box in the project plan's architecture diagram (Figure 1). Wraps a model
bundle produced by `python -m src.pipeline --dataset <name> --save-model`
(src/pipeline.py's optional Step 8), which pickles the fitted
ConfidenceWeightedStackingEnsemble alongside its scaler path, task type, and
expected feature column order.

Models are loaded lazily per dataset and cached in-process, so one process
can serve creditcard, mammography, and yeast_multiclass simultaneously
without restarting.

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

    curl -X POST http://localhost:8000/predict/creditcard \
        -H "Content-Type: application/json" \
        -d '{"features": {"Time": 0, "V1": -1.36, ..., "Amount": 149.62}}'
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("serving")

MODELS_DIR = Path("models")

app = FastAPI(
    title="ARS-Stack Inference API",
    description="Serves the confidence-weighted stacking ensemble trained by src/pipeline.py.",
    version="1.0.0",
)

_model_cache: Dict[str, dict] = {}


class PredictRequest(BaseModel):
    features: Dict[str, float]


class PredictResponse(BaseModel):
    dataset: str
    task_type: str
    prediction: int
    probabilities: Dict[str, float]


def _load_bundle(dataset_name: str) -> dict:
    """Load (and cache) the model bundle saved by `pipeline.py --save-model`."""
    if dataset_name in _model_cache:
        return _model_cache[dataset_name]

    model_path = MODELS_DIR / f"{dataset_name}.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No saved model for '{dataset_name}' at {model_path}. Train and "
                f"export one first: python -m src.pipeline --dataset {dataset_name} --save-model"
            ),
        )

    bundle = joblib.load(model_path)
    scaler = None
    if bundle.get("scaler_path"):
        scaler_path = Path(bundle["scaler_path"])
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
    bundle["scaler"] = scaler
    _model_cache[dataset_name] = bundle
    logger.info("Loaded model bundle for '%s' from %s", dataset_name, model_path)
    return bundle


def _vectorize(bundle: dict, features: Dict[str, float]) -> np.ndarray:
    """Turns a {feature_name: value} request payload into the (1, n_features)
    row the model expects, in the exact column order used at training time,
    then applies the same StandardScaler fit during preprocessing — to ONLY
    the columns it was fit on (creditcard scales just Time/Amount; the V1-V28
    PCA components are left as-is)."""
    columns = bundle["feature_columns"]
    missing = [c for c in columns if c not in features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required feature(s): {missing}")

    row = pd.DataFrame([[features[c] for c in columns]], columns=columns, dtype=float)
    scaler = bundle["scaler"]
    if scaler is not None:
        scaled_cols = list(getattr(scaler, "feature_names_in_", columns))
        row[scaled_cols] = scaler.transform(row[scaled_cols])
    return row.to_numpy()


@app.get("/health")
def health():
    return {"status": "ok", "cached_models": list(_model_cache.keys())}


@app.get("/models/{dataset_name}")
def model_info(dataset_name: str):
    bundle = _load_bundle(dataset_name)
    return {
        "dataset": bundle["dataset"],
        "task_type": bundle["task_type"],
        "n_features": len(bundle["feature_columns"]),
        "feature_columns": bundle["feature_columns"],
        "combiner": getattr(bundle["model"], "combiner", None),
        "base_learner_weights": bundle["model"].weights_,
        "decision_threshold": getattr(bundle["model"], "threshold_", None) if bundle["task_type"] == "binary" else None,
    }


@app.post("/predict/{dataset_name}", response_model=PredictResponse)
def predict(dataset_name: str, request: PredictRequest):
    bundle = _load_bundle(dataset_name)
    model = bundle["model"]
    task_type = bundle["task_type"]

    X = _vectorize(bundle, request.features)
    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] == 2 and task_type == "binary":
        proba_row = proba[0]
        class_labels = ["0", "1"]
    else:
        proba_row = proba[0]
        classes = getattr(model, "classes_", None)
        class_labels = [str(c) for c in (classes if classes is not None else range(len(proba_row)))]

    pred = model.predict(X)[0]

    return PredictResponse(
        dataset=dataset_name,
        task_type=task_type,
        prediction=int(pred),
        probabilities={label: float(p) for label, p in zip(class_labels, proba_row)},
    )
