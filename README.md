# MLOps Lab: MLflow + FastAPI + Docker

Kursun 7-11. bölümlerini tek küçük projede uygulamalı pekiştirir.

## 1. Kurulum
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Deneyleri çalıştır (Bölüm 7-9)
```bash
python train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
http://127.0.0.1:5000 adresinde:
- **Experiments > iris-lab:** 9 run'ı seçip *Compare* ile karşılaştır. `n_estimators=1, max_depth=1` bilerek kötü.
- **Models > iris-classifier:** en iyi model `production` etiketiyle kayıtlı.

## 3. API'yi yerelde çalıştır (Bölüm 10)
```bash
uvicorn app:app --reload
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
Swagger arayüzü: http://127.0.0.1:8000/docs

## 4. Docker (Bölüm 11)
```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```
Her istek `logs/predictions.jsonl` dosyasına yazılır (container içinde). Dışarı almak için:
`docker run -p 8000:8000 -v "$(pwd)/logs:/app/logs" iris-api`

## Sıradaki denemeler
- **Bölüm 12:** `.github/workflows/ci.yml` ekle: push'ta `pip install`, `python train.py`, `docker build` çalışsın.
- **Bölüm 14:** `logs/predictions.jsonl` içindeki girdi dağılımını eğitim verisiyle karşılaştıran bir drift script'i yaz (Evidently ya da elle ortalama/std kontrolü). Önce `curl` ile bilerek kaydırılmış değerler gönder.
- **Bölüm 15:** Drift varsa `train.py`'yi yeniden çalıştıran ve yeni sürümü Registry'ye düşüren bir script ekle.
- **Bölüm 6:** `git init`, sonra veriyi DVC ile versiyonla (Iris yerine kendi CSV'ni kullan).

## Notlar
- Yeni MLflow sürümleri sklearn modellerini `skops` ile kaydediyor. `train.py` içindeki `TRUSTED` listesi bu yüzden var; kendi eğittiğin modele güvendiğin için sorun değil.
- Eğitim ve servis ortamında scikit-learn sürümleri farklıysa uyarı alırsın. `pip freeze | grep scikit-learn` ile bakıp `requirements-serve.txt` içine aynı sürümü yazabilirsin.
