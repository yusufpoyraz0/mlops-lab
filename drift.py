"""Bölüm 14: logs/predictions.jsonl içindeki canlı girdileri eğitim verisiyle karşılaştır.

Çıkış kodu: 0 = drift yok, 1 = drift var (Bölüm 15'te yeniden eğitimi tetiklemek için kullanılır).
"""
import sys

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris

LOG_FILE = "logs/predictions.jsonl"
P_THRESHOLD = 0.01   # p-değeri bunun altındaysa dağılım değişmiş say
MIN_ROWS = 20        # çok az istekle karar verme

ref, _ = load_iris(as_frame=True, return_X_y=True)
ref = ref.rename(columns=lambda c: c.replace(" (cm)", "").replace(" ", "_"))

live = pd.read_json(LOG_FILE, lines=True)
if len(live) < MIN_ROWS:
    print(f"Yetersiz veri ({len(live)}/{MIN_ROWS}). Önce daha fazla istek gönder.")
    sys.exit(0)

drifted = False
for col in ref.columns:
    stat, p = ks_2samp(ref[col], live[col])
    flag = "DRIFT" if p < P_THRESHOLD else "ok"
    drifted |= p < P_THRESHOLD
    print(f"{col:<14} eğitim ort={ref[col].mean():.2f}  canlı ort={live[col].mean():.2f}  p={p:.4f}  {flag}")

print("\nSonuç:", "DRIFT VAR" if drifted else "drift yok")
sys.exit(1 if drifted else 0)
