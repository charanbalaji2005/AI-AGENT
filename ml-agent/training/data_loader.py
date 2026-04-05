"""
data_loader.py — Load, clean, and prepare your CSV dataset for training.
Replace data/dataset.csv with your own file (same columns: text, label).
"""

import os
import re
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "../models")
TEXT_COL    = "text"    # change if your CSV column is named differently
LABEL_COL   = "label"  # change if your CSV column is named differently
TEST_SIZE   = 0.2
RANDOM_SEED = 42


def clean_text(text: str) -> str:
    """Lowercase, remove special chars, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    """
    Load CSV → clean → encode labels → train/test split.
    Returns: X_train, X_test, y_train, y_test, label_encoder
    """
    print(f"📂 Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Validate columns
    assert TEXT_COL  in df.columns, f"Column '{TEXT_COL}' not found in CSV"
    assert LABEL_COL in df.columns, f"Column '{LABEL_COL}' not found in CSV"

    # Drop nulls and duplicates
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df.drop_duplicates(subset=[TEXT_COL])

    print(f"✅ Loaded {len(df)} samples | {df[LABEL_COL].nunique()} classes")
    print(f"📊 Class distribution:\n{df[LABEL_COL].value_counts().to_string()}\n")

    # Clean text
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

    # Encode labels
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df[LABEL_COL])

    # Save label encoder
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Save class names to JSON for the API
    class_info = {
        "classes": le.classes_.tolist(),
        "num_classes": len(le.classes_),
    }
    with open(os.path.join(MODELS_DIR, "class_info.json"), "w") as f:
        json.dump(class_info, f, indent=2)

    print(f"🏷️  Classes: {le.classes_.tolist()}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL].tolist(),
        df["label_enc"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label_enc"],
    )

    print(f"📦 Train: {len(X_train)} | Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test, le


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = load_dataset()
    print("Data loading complete.")
