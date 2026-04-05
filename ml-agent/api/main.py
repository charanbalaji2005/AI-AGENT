"""
api/main.py — FastAPI server that serves your trained classifier.

Endpoints:
  GET  /                    → health check
  GET  /classes             → list all label classes
  GET  /model/info          → model metadata + accuracy
  POST /predict             → classify a single text
  POST /predict/batch       → classify multiple texts at once
  POST /train               → trigger re-training from the API (async)

Usage:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
  OR
  python api/main.py
"""

import os
import json
import pickle
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "../training/train.py")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="🤖 Domain Text Classifier API",
    description=(
        "Train and serve a custom ML classifier on your domain CSV dataset. "
        "Built with scikit-learn + FastAPI. CPU-friendly."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model State ───────────────────────────────────────────────────────────────
_model_state = {
    "pipeline": None,
    "label_encoder": None,
    "classes": [],
    "model_name": None,
    "metrics": {},
    "loaded_at": None,
}
_training_status = {"running": False, "last_output": "", "error": ""}


def load_model():
    """Load the best trained model from disk into memory."""
    best_path = os.path.join(MODELS_DIR, "best_model.json")
    if not os.path.exists(best_path):
        return False

    with open(best_path) as f:
        best = json.load(f)["best_model"]

    pipeline_path = os.path.join(MODELS_DIR, f"{best}_pipeline.pkl")
    le_path       = os.path.join(MODELS_DIR, "label_encoder.pkl")
    class_path    = os.path.join(MODELS_DIR, "class_info.json")
    metrics_path  = os.path.join(MODELS_DIR, f"{best}_metrics.json")

    if not os.path.exists(pipeline_path):
        return False

    with open(pipeline_path, "rb") as f:
        _model_state["pipeline"] = pickle.load(f)

    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            _model_state["label_encoder"] = pickle.load(f)

    if os.path.exists(class_path):
        with open(class_path) as f:
            info = json.load(f)
            _model_state["classes"] = info["classes"]

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _model_state["metrics"] = json.load(f)

    _model_state["model_name"] = best
    _model_state["loaded_at"]  = time.strftime("%Y-%m-%dT%H:%M:%S")
    return True


# Load on startup
@app.on_event("startup")
def startup():
    loaded = load_model()
    if loaded:
        print(f"✅ Model loaded: {_model_state['model_name']}")
    else:
        print("⚠️  No trained model found. POST /train first.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., example="How does binary search work?", min_length=2)
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Return top-K predictions")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., example=["What is a REST API?", "Explain quicksort"])
    top_k: Optional[int] = Field(1, ge=1, le=10)

class TrainRequest(BaseModel):
    model: Optional[str] = Field("logreg", description="Model to train: logreg|svm|naive_bayes|random_forest|gradient_boost|all")


# ── Helpers ───────────────────────────────────────────────────────────────────
def require_model():
    if _model_state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Run POST /train first, then /model/reload."
        )

def predict_text(text: str, top_k: int = 1):
    pipeline = _model_state["pipeline"]
    classes  = _model_state["classes"]

    # Get decision scores if available, else use predict
    clf = pipeline.named_steps["clf"]
    tfidf = pipeline.named_steps["tfidf"]
    X_vec = tfidf.transform([text])

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_vec)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        predictions = [
            {"label": classes[i], "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_vec)[0]
        if scores.ndim == 0:
            scores = np.array([scores])
        # Softmax-like normalisation for display
        e = np.exp(scores - scores.max())
        probs = e / e.sum()
        top_indices = np.argsort(probs)[::-1][:top_k]
        predictions = [
            {"label": classes[i], "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]
    else:
        label_idx = clf.predict(X_vec)[0]
        predictions = [{"label": classes[label_idx], "confidence": 1.0}]

    return predictions


def run_training(model: str):
    _training_status["running"] = True
    _training_status["error"] = ""
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT, "--model", model],
            capture_output=True, text=True, timeout=600
        )
        _training_status["last_output"] = result.stdout[-3000:]
        if result.returncode != 0:
            _training_status["error"] = result.stderr[-1000:]
        else:
            load_model()  # hot-reload after training
    except Exception as e:
        _training_status["error"] = str(e)
    finally:
        _training_status["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    model_loaded = _model_state["pipeline"] is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_name": _model_state.get("model_name"),
        "classes": _model_state.get("classes"),
        "docs": "/docs",
    }


@app.get("/classes", tags=["Model"])
def get_classes():
    """Return all label classes the model can predict."""
    require_model()
    return {
        "classes": _model_state["classes"],
        "count": len(_model_state["classes"]),
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata about the loaded model."""
    require_model()
    return {
        "model_name":  _model_state["model_name"],
        "classes":     _model_state["classes"],
        "num_classes": len(_model_state["classes"]),
        "metrics":     _model_state["metrics"],
        "loaded_at":   _model_state["loaded_at"],
    }


@app.post("/model/reload", tags=["Model"])
def reload_model():
    """Reload the model from disk (after re-training)."""
    loaded = load_model()
    if not loaded:
        raise HTTPException(status_code=404, detail="No trained model found on disk.")
    return {"status": "reloaded", "model_name": _model_state["model_name"]}


@app.post("/predict", tags=["Predict"])
def predict(req: PredictRequest):
    """
    Classify a single text snippet.
    Returns top-K label predictions with confidence scores.
    """
    require_model()
    start = time.time()
    predictions = predict_text(req.text, req.top_k)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "text":        req.text,
        "predictions": predictions,
        "top_label":   predictions[0]["label"],
        "confidence":  predictions[0]["confidence"],
        "latency_ms":  elapsed_ms,
        "model_used":  _model_state["model_name"],
    }


@app.post("/predict/batch", tags=["Predict"])
def predict_batch(req: BatchPredictRequest):
    """
    Classify multiple texts at once.
    Returns predictions for each text.
    """
    require_model()
    if len(req.texts) > 500:
        raise HTTPException(status_code=400, detail="Max 500 texts per batch.")

    start = time.time()
    results = []
    for text in req.texts:
        preds = predict_text(text, req.top_k)
        results.append({
            "text":       text,
            "top_label":  preds[0]["label"],
            "confidence": preds[0]["confidence"],
            "predictions": preds,
        })
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "count":      len(results),
        "results":    results,
        "latency_ms": elapsed_ms,
        "model_used": _model_state["model_name"],
    }


@app.post("/train", tags=["Training"])
def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger model training in the background.
    Check /train/status for progress.
    """
    if _training_status["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    valid_models = ["logreg", "svm", "naive_bayes", "random_forest", "gradient_boost", "all"]
    if req.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {valid_models}")

    background_tasks.add_task(run_training, req.model)
    return {
        "status":  "started",
        "message": f"Training '{req.model}' in background. Check /train/status.",
    }


@app.get("/train/status", tags=["Training"])
def training_status():
    """Check if training is running and see the latest output."""
    return {
        "running":     _training_status["running"],
        "last_output": _training_status["last_output"][-2000:],
        "error":       _training_status["error"],
        "model_loaded": _model_state["model_name"],
    }


# ── Dev server entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
