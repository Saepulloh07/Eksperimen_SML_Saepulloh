# File: 7.inference.py (FINAL — SUDAH FIX 13 FITUR + ERROR HANDLING)

import time
import joblib
import pickle
import numpy as np
import requests
import os
from datetime import datetime

# Path sesuai struktur kamu
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "Membangun_model", "best_model_tuned.pkl")
SCALER_PATH = os.path.join(ROOT, "Membangun_model", "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(ROOT, "preprosessing", "data", "output", "feature_names.pkl")

EXPORTER_URL = "http://localhost:8000"

class WineQualityPredictor:
    def __init__(self):
        print("Loading model, scaler, dan feature names...")
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            self.feature_names = pickle.load(f)
        print(f"SIAP! Model + Scaler + {len(self.feature_names)} fitur berhasil dimuat.")
        print(f"Fitur yang dibutuhkan: {self.feature_names}")

    def predict(self, features_list):
        # Validasi jumlah fitur
        for i, row in enumerate(features_list):
            if len(row) != len(self.feature_names):
                raise ValueError(f"Sample {i}: Expected {len(self.feature_names)} features, got {len(row)}")

        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled).astype(int)
        probas = self.model.predict_proba(X_scaled)

        # Kirim ke Prometheus Exporter
        try:
            requests.post(f"{EXPORTER_URL}/record_prediction", json={
                "X": X_scaled.tolist(),
                "y_pred": y_pred.tolist(),
                "probabilities": probas.tolist()
            }, timeout=3)
            print("Metrics berhasil dikirim ke Prometheus!")
        except Exception as e:
            print(f"Exporter tidak aktif: {e}")

        # Return hasil prediksi
        results = []
        for i, (pred, proba) in enumerate(zip(y_pred, probas)):
            results.append({
                "id": i,
                "prediction": int(pred),
                "label": "Good Wine" if pred >= 6 else "Bad Wine",  # biasanya quality 6+ = good
                "confidence": round(float(np.max(proba)), 4),
                "all_probabilities": {f"class_{k}": round(float(p), 4) for k, p in enumerate(proba)},
                "timestamp": datetime.now().isoformat()
            })
        return results

# ============================================================
# CONTOH DATA YANG BENAR DENGAN 13 FITUR
# ============================================================

if __name__ == "__main__":
    predictor = WineQualityPredictor()

    # PAKAI DATA DENGAN 13 FITUR (sesuai feature_names.pkl kamu)
    # Contoh: ambil dari X_test_scaled.npy kalau ada, atau buat dummy yang valid
    dummy_data_13_features = [
        [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 5.0, 1.0],  # 13 fitur
        [7.8, 0.88, 0.00, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.20, 0.68, 9.8, 5.0, 1.0],
        [6.3, 0.30, 0.34, 1.6, 0.049, 14.0, 132.0, 0.9940, 3.30, 0.49, 9.5, 6.0, 1.0],
    ]

    print("\n" + "="*60)
    print("MELAKUKAN PREDIKSI DENGAN 13 FITUR")
    print("="*60)

    try:
        results = predictor.predict(dummy_data_13_features)
        for r in results:
            print(f"→ {r['label']} | Confidence: {r['confidence']} | Classes: {r['all_probabilities']}")
    except Exception as e:
        print(f"ERROR: {e}")

    print(f"\nMonitoring aktif!")
    print(f"→ Prometheus: http://localhost:8000/metrics")
    print(f"→ Health check: http://localhost:8000/health")
    print("="*60)

    # Keep alive
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nInference dihentikan.")