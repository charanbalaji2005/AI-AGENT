# 🤖 Domain Text Classifier — ML Agent

A complete, production-ready ML pipeline that trains a **text classifier on your own CSV dataset** and serves it via a **FastAPI REST API**. Fully CPU-friendly.

---

## Project Structure

```
ml-agent/
├── data/
│   └── dataset.csv          ← YOUR dataset goes here
│
├── training/
│   ├── data_loader.py       ← load + clean + split CSV
│   ├── train.py             ← train TF-IDF + ML model
│   └── evaluate.py          ← detailed metrics + confusion matrix
│
├── models/                  ← auto-created after training
│   ├── logreg_pipeline.pkl
│   ├── label_encoder.pkl
│   ├── class_info.json
│   └── best_model.json
│
├── api/
│   └── main.py              ← FastAPI server
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd ml-agent
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare your dataset

Your CSV file must have at least two columns:

```csv
text,label
"How do I reverse a linked list?",data_structures
"What is gradient descent?",machine_learning
"Explain REST APIs",web_development
...
```

- Column names: `text` and `label` (change in `training/data_loader.py` if different)
- Minimum recommended: **50+ samples**, at least **10 per class**
- More data = better accuracy

Replace `data/dataset.csv` with your file.

### 3. Train the model

```bash
# Train default (Logistic Regression — fast, accurate)
python training/train.py

# Train a specific model
python training/train.py --model svm
python training/train.py --model random_forest
python training/train.py --model naive_bayes
python training/train.py --model gradient_boost

# Train ALL models and auto-select the best
python training/train.py --model all
```

**Available models:**

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| `logreg` | ⚡ Fast | ⭐⭐⭐⭐ | Best default choice |
| `svm` | ⚡ Fast | ⭐⭐⭐⭐⭐ | Often top performer |
| `naive_bayes` | ⚡⚡ Very fast | ⭐⭐⭐ | Good baseline |
| `random_forest` | 🐢 Slow | ⭐⭐⭐⭐ | Robust, interpretable |
| `gradient_boost` | 🐢 Slow | ⭐⭐⭐⭐ | Strong on small data |

### 4. Evaluate

```bash
python training/evaluate.py
# or specify model:
python training/evaluate.py --model svm
```

Shows accuracy, F1, precision, recall, per-class breakdown, and confusion matrix.

### 5. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

## API Endpoints

### Health Check
```
GET /
```

### Predict a single text
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I implement a binary search?", "top_k": 3}'
```

Response:
```json
{
  "text": "How do I implement a binary search?",
  "top_label": "algorithms",
  "confidence": 0.8721,
  "predictions": [
    {"label": "algorithms", "confidence": 0.8721},
    {"label": "data_structures", "confidence": 0.0812},
    {"label": "machine_learning", "confidence": 0.0467}
  ],
  "latency_ms": 4.2,
  "model_used": "logreg"
}
```

### Batch predict
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is a REST API?", "Explain quicksort"], "top_k": 1}'
```

### Trigger re-training via API
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"model": "svm"}'
```

### Check training status
```
GET /train/status
```

### Model info
```
GET /model/info
```

---

## Customising

### Add more classes
Just add more rows to your CSV with new label values — the pipeline auto-discovers all classes.

### Change feature extraction
Edit `TFIDF_CONFIG` in `training/train.py`:
```python
TFIDF_CONFIG = dict(
    ngram_range=(1, 3),   # include trigrams
    max_features=50_000,
    ...
)
```

### Add your own model
In `training/train.py`:
```python
from sklearn.neighbors import KNeighborsClassifier

MODEL_REGISTRY["knn"] = KNeighborsClassifier(n_neighbors=5)
```

---

## Performance Tips (CPU)

- Use `logreg` or `svm` — fastest training, top accuracy for text
- Keep `max_features=30000` in TF-IDF (good balance)
- Use `n_jobs=-1` in Random Forest to use all CPU cores
- For 10k+ samples, `svm` with `LinearSVC` is usually best

---

## License
MIT
