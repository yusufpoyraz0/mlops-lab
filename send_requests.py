import sys

import httpx
from sklearn.datasets import load_iris

mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
X, _ = load_iris(as_frame=True, return_X_y=True)
rows = X.sample(30, random_state=1).values.tolist()

for r in rows:
    if mode == "shifted":
        r[0] += 1.5   # sepal_length'i kaydır
        r[2] += 2.0   # petal_length'i kaydır
    body = dict(zip(["sepal_length", "sepal_width", "petal_length", "petal_width"], r))
    httpx.post("http://127.0.0.1:8000/predict", json=body)

print("30 istek gonderildi:", mode)