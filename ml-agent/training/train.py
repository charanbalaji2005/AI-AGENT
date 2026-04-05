"""
train.py — Train a text classifier on your domain CSV dataset.

Pipeline:
  CSV → TF-IDF vectorizer → Model (LogReg / SVM / RandomForest / XGBoost)
  → evaluate → save model + vectorizer → ready for FastAPI

Usage:
  python training/train.py
  python training/train.py --model svm
  python training/train.py --model all    ← trains all, picks best
"""

import os
import json
import pickle
import argparse
import time
import numpy as np

from sklearn.pipeline         import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import LinearSVC
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes      import MultinomialNB
from sklearn.metrics          import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.model_selection  import cross_val_score

# Resolve paths relative to this file
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
os.makedirs(MODELS_DIR, exist_ok=True)

import sys
sys.path.insert(0, BASE_DIR)
from data_loader import load_dataset


# ── Available model configs ──────────────────────────────────────────────────
MODEL_REGISTRY = {
    "logreg": LogisticRegression(
        max_iter=1000, C=5.0, solver="lbfgs"
    ),
    "svm": LinearSVC(
        max_iter=2000, C=1.0, dual=True
    ),
    "naive_bayes": MultinomialNB(alpha=0.1),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    ),
    "gradient_boost": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
}

# ── TF-IDF config (CPU-efficient, works well on domain text) ─────────────────
TFIDF_CONFIG = dict(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=30_000,
    sublinear_tf=True,
    min_df=1,
    analyzer="word",
    token_pattern=r"\w{2,}",  # skip single-char tokens
)


def build_pipeline(model_name: str) -> Pipeline:
    clf = MODEL_REGISTRY[model_name]
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        ("clf",   clf),
    ])


def evaluate(pipeline, X_test, y_test, le, model_name):
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=3,
    )
    print(f"\n{'='*55}")
    print(f"  Model : {model_name.upper()}")
    print(f"  Accuracy : {acc:.4f}  |  F1 (weighted) : {f1:.4f}")
    print(f"{'='*55}")
    print(report)
    return acc, f1


def save_model(pipeline, model_name, metrics):
    path = os.path.join(MODELS_DIR, f"{model_name}_pipeline.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)

    # Save metrics alongside
    meta = {
        "model_name": model_name,
        "accuracy":   round(metrics["accuracy"], 4),
        "f1_weighted": round(metrics["f1"], 4),
        "saved_path": path,
    }
    with open(os.path.join(MODELS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Saved → {path}")
    return path


def save_best_model_pointer(model_name):
    """Write a pointer so the API always loads the best model."""
    with open(os.path.join(MODELS_DIR, "best_model.json"), "w") as f:
        json.dump({"best_model": model_name}, f, indent=2)
    print(f"\n🏆 Best model set to: {model_name}")


def train_one(model_name, X_train, X_test, y_train, y_test, le, cv=True):
    print(f"\n🚀 Training [{model_name}] ...")
    pipeline = build_pipeline(model_name)

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"⏱️  Training time: {elapsed:.2f}s")

    # Cross-validation on training set
    if cv:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        print(f"📈 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    acc, f1 = evaluate(pipeline, X_test, y_test, le, model_name)
    save_model(pipeline, model_name, {"accuracy": acc, "f1": f1})
    return acc, f1


def train_all(X_train, X_test, y_train, y_test, le):
    results = {}
    for name in MODEL_REGISTRY:
        acc, f1 = train_one(name, X_train, X_test, y_train, y_test, le, cv=False)
        results[name] = {"accuracy": acc, "f1": f1}

    # Pick best by F1
    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\n{'='*55}")
    print("  LEADERBOARD")
    print(f"{'='*55}")
    for name, m in sorted(results.items(), key=lambda x: -x[1]["f1"]):
        flag = " ← BEST" if name == best else ""
        print(f"  {name:<20} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}{flag}")
    save_best_model_pointer(best)
    return best


def main():
    parser = argparse.ArgumentParser(description="Train domain text classifier")
    parser.add_argument(
        "--model",
        default="logreg",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Which model to train (default: logreg)"
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, le = load_dataset()

    if args.model == "all":
        best = train_all(X_train, X_test, y_train, y_test, le)
        print(f"\n✅ All models trained. Best: {best}")
    else:
        train_one(args.model, X_train, X_test, y_train, y_test, le)
        save_best_model_pointer(args.model)
        print(f"\n✅ Training complete.")


if __name__ == "__main__":
    main()
