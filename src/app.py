#!/usr/bin/env python3
"""
src/app.py
----------
FastAPI API exposing:
  - GET /health
  - POST /classify

Defaults to BERT and auto-loads the most recent run from ../experiments_bert/.
Works with infer.py inside the same src/ directory.
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

# -----------------------------------
# Import infer helpers from same dir
# -----------------------------------
from infer import (
    load_sgd,
    predict_sgd,
    load_bert,
    predict_bert,
)

# -----------------------------------
# App setup
# -----------------------------------
app = FastAPI(title="Ticket Classifier API", version="1.1.0")

# Default to BERT
DEFAULT_MODEL = os.getenv("MODEL_TYPE", "bert").lower()  # "bert" or "sgd"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.3"))

# Experiments roots (relative to src/)
EXPERIMENTS_DIR_BERT = Path("../experiments_bert")
EXPERIMENTS_DIR_SGD = Path("../experiments_sgd")

# Optional explicit model dir via env (otherwise we will auto-pick latest)
ENV_MODEL_DIR = os.getenv("MODEL_DIR", "").strip() or None


# -----------------------------------
# Helpers to find latest run
# -----------------------------------
def _latest_run_dir(root: Path) -> Optional[Path]:
    """Return most recent run-* directory under root, or None."""
    if not root.exists():
        return None
    runs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("run-")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]

def resolve_default_model_dir(model_type: str) -> Optional[str]:
    """
    For 'bert': return latest run dir under experiments_bert/.
    For 'sgd' : return path to model.pkl under latest run under experiments_sgd/.
    Returns None if not found.
    """
    if model_type == "bert":
        latest = _latest_run_dir(EXPERIMENTS_DIR_BERT)
        return str(latest) if latest else None
    elif model_type == "sgd":
        latest = _latest_run_dir(EXPERIMENTS_DIR_SGD)
        if not latest:
            return None
        pkl = latest / "model.pkl"
        return str(pkl) if pkl.exists() else None
    return None


# -----------------------------------
# Model cache (for quick reuse)
# -----------------------------------
class Registry:
    sgd = {"loaded": False, "model": None, "classes": None, "path": None}
    bert = {"loaded": False, "tokenizer": None, "model": None, "labels": None, "path": None}

REG = Registry()


@app.on_event("startup")
def preload_model():
    """
    Preload default model on startup to avoid cold starts.
    If MODEL_DIR env is not set, auto-pick the latest run for the chosen model type.
    """
    try:
        model_dir = ENV_MODEL_DIR or resolve_default_model_dir(DEFAULT_MODEL)

        if not model_dir:
            print(f"⚠️ No {DEFAULT_MODEL.upper()} model found to preload (no runs yet).")
            return

        if DEFAULT_MODEL == "sgd":
            model, classes = load_sgd(model_dir)
            REG.sgd.update({"loaded": True, "model": model, "classes": classes, "path": model_dir})
        elif DEFAULT_MODEL == "bert":
            tokenizer, model, labels = load_bert(model_dir)
            REG.bert.update({"loaded": True, "tokenizer": tokenizer, "model": model, "labels": labels, "path": model_dir})

        print(f"✅ Preloaded default {DEFAULT_MODEL.upper()} model from: {model_dir}")

    except Exception as e:
        print(f"⚠️ Warning: Failed to preload {DEFAULT_MODEL}: {e}")


# -----------------------------------
# Request/Response schemas
# -----------------------------------
class ClassifyRequest(BaseModel):
    title: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Ticket description")
    answer: Optional[str] = ""
    type: Optional[str] = ""
    tag: Optional[str] = ""
    model_type: Optional[Literal["sgd", "bert"]] = None
    model_dir: Optional[str] = None
    threshold: Optional[float] = None


class ClassifyResponse(BaseModel):
    predicted_category: str
    confidence: float
    raw: Dict[str, Any]


# -----------------------------------
# Routes
# -----------------------------------
@app.get("/health")
def health():
    latest_bert = _latest_run_dir(EXPERIMENTS_DIR_BERT)
    latest_sgd = _latest_run_dir(EXPERIMENTS_DIR_SGD)
    return {
        "status": "ok",
        "default_model_type": DEFAULT_MODEL,
        "env_model_dir": ENV_MODEL_DIR,
        "bert_latest_run": str(latest_bert) if latest_bert else None,
        "sgd_latest_run": str(latest_sgd / "model.pkl") if latest_sgd and (latest_sgd / "model.pkl").exists() else None,
        "bert_loaded": REG.bert["loaded"],
        "bert_loaded_from": REG.bert["path"],
        "sgd_loaded": REG.sgd["loaded"],
        "sgd_loaded_from": REG.sgd["path"],
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    model_type = (req.model_type or DEFAULT_MODEL).lower()
    threshold = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD

    if not req.title or not req.description:
        raise HTTPException(status_code=400, detail="Both title and description are required.")

    # Resolve model_dir priority: request override -> env -> latest
    if req.model_dir:
        model_dir = req.model_dir
    elif ENV_MODEL_DIR:
        model_dir = ENV_MODEL_DIR
    else:
        model_dir = resolve_default_model_dir(model_type)
        if not model_dir:
            raise HTTPException(status_code=404, detail=f"No {model_type.upper()} model found (no runs yet).")

    # --- SGD prediction ---
    if model_type == "sgd":
        if not REG.sgd["loaded"] or REG.sgd["path"] != model_dir:
            model, classes = load_sgd(model_dir)
            REG.sgd.update({"loaded": True, "model": model, "classes": classes, "path": model_dir})

        label, conf, scores, labels = predict_sgd(REG.sgd["model"], REG.sgd["classes"], req.model_dump(), threshold)

    # --- BERT prediction ---
    elif model_type == "bert":
        if not REG.bert["loaded"] or REG.bert["path"] != model_dir:
            tokenizer, model, labels = load_bert(model_dir)
            REG.bert.update({"loaded": True, "tokenizer": tokenizer, "model": model, "labels": labels, "path": model_dir})

        label, conf, scores, labels = predict_bert(REG.bert["tokenizer"], REG.bert["model"], REG.bert["labels"], req.model_dump(), threshold)

    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'sgd' or 'bert'.")

    final_label = "General Inquiry" if conf < threshold else label
    return ClassifyResponse(
        predicted_category=final_label,
        confidence=round(conf, 4),
        #raw={"scores": scores, "labels": labels, "model_type": model_type, "model_dir": model_dir},
    )


# -----------------------------------
# Run locally
# -----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
