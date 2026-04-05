"""
evaluate.py — Deep evaluation of your trained model.
Run after training to get detailed metrics, per-class breakdown, and confusion matrix.

Usage:
  python training/evaluate.py
  python training/evaluate.py --model svm
"""

import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
sys.path.insert(0, BASE_DIR)
from data_loader import load_dataset


def evaluate_model(model_name: str):
    pipeline_path = os.path.join(MODELS_DIR, f"{model_name}_pipeline.pkl")
    if not os.path.exists(pipeline_path):
        print(f"❌ Model '{model_name}' not found. Train it first.")
        return

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    _, X_test, _, y_test, _ = load_dataset()

    y_pred = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Model Evaluation: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")

    print(f"\n── Per-Class Report ──")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

    print(f"\n── Confusion Matrix ──")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())

    # Save evaluation results
    eval_results = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "per_class": {},
    }
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    for cls in le.classes_:
        if cls in report:
            eval_results["per_class"][cls] = {k: round(v, 4) for k, v in report[cls].items()}

    out_path = os.path.join(MODELS_DIR, f"{model_name}_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n💾 Evaluation saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model name to evaluate")
    args = parser.parse_args()

    if args.model:
        evaluate_model(args.model)
    else:
        best_path = os.path.join(MODELS_DIR, "best_model.json")
        if os.path.exists(best_path):
            with open(best_path) as f:
                best = json.load(f)["best_model"]
            evaluate_model(best)
        else:
            print("No trained model found. Run training/train.py first.")
