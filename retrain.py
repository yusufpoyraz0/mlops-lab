"""Bölüm 15: Drift varsa modeli yeniden eğit."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

LOG_FILE = Path("logs/predictions.jsonl")
DRIFT_EXIT_CODE = 10

drift = subprocess.run([sys.executable, "drift.py"])

if drift.returncode == 0:
    print("\nDrift yok, yeniden egitime gerek yok.")
    sys.exit(0)
if drift.returncode != DRIFT_EXIT_CODE:
    print("\ndrift.py beklenmeyen bir hatayla bitti, yeniden egitim yapilmadi.")
    sys.exit(1)

print("\nDrift var, model yeniden egitiliyor...")
subprocess.run([sys.executable, "train.py"], check=True)

# Eski logları arşivle ki bir sonraki kontrol taze veriyle başlasın
archive = LOG_FILE.with_name(f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
shutil.move(str(LOG_FILE), str(archive))
print(f"Loglar arsivlendi: {archive}")
print("Yeni modeli canliya almak icin: docker build -t iris-api .  (API'yi de yeniden baslat)")
