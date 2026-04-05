import requests

BASE = "http://127.0.0.1:8000"

# Health check
print(requests.get(f"{BASE}/").json())

# Single predict
resp = requests.post(f"{BASE}/predict", json={
    "text": "How do I implement quicksort?",
    "top_k": 3
})
print(resp.json())

# Batch predict
resp = requests.post(f"{BASE}/predict/batch", json={
    "texts": [
        "What is a REST API?",
        "Explain gradient descent",
        "How does DNS work?",
        "What is a binary tree?"
    ],
    "top_k": 1
})
for r in resp.json()["results"]:
    print(f"  {r['text'][:40]:<40} → {r['top_label']} ({r['confidence']})")