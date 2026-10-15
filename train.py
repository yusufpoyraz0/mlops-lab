"""Bölüm 7-9: Birden fazla deney çalıştır, MLflow'da izle, en iyisini Registry'ye kaydet."""
import itertools

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "iris-classifier"
# Yeni MLflow sürümleri sklearn modellerini skops ile kaydediyor; kendi eğittiğimiz ağaç tipine güveniyoruz.
TRUSTED = ["sklearn.tree._tree.Tree"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-lab")

X, y = load_iris(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Hiperparametre ızgarası. (1, 1) bilerek kötü bir deney: izlemede nasıl göründüğüne bak.
grid = itertools.product([1, 10, 100], [1, 3, None])

best = {"f1": -1.0, "run_id": None, "model": None}

for n_estimators, max_depth in grid:
    with mlflow.start_run(run_name=f"rf-n{n_estimators}-d{max_depth}") as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        ).fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=TRUSTED)
        print(f"n_estimators={n_estimators!s:>3} max_depth={max_depth!s:>4} -> acc={acc:.3f} f1={f1:.3f}")

        if f1 > best["f1"]:
            best = {"f1": f1, "run_id": run.info.run_id, "model": model}

# En iyi modeli Registry'ye kaydet ve "production" etiketini ver (Bölüm 9)
version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version.version)
print(f"\nEn iyi run: {best['run_id']} (f1={best['f1']:.3f}) -> {MODEL_NAME} v{version.version} [production]")

# Docker imajına girecek yerel kopya (Bölüm 10-11)
mlflow.sklearn.save_model(best["model"], "model", skops_trusted_types=TRUSTED)
print("Model 'model/' klasörüne kaydedildi.")
