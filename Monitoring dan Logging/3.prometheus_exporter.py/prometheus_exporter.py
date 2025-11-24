# File: Monitoring dan Logging/3.Prometheus_exporter/prometheus_exporter.py

import time
import psutil
import joblib
import pickle
import numpy as np
from flask import Flask, Response, request, jsonify
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST, Info
)
import threading
import os

# ============================================================
# PATH YANG BENAR 100% SESUAI FOLDER KAMU (prosessing/data/output)
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
TRUE_ROOT = os.path.dirname(PARENT_DIR)

MODEL_PATH         = os.path.join(TRUE_ROOT, "Membangun_model", "best_model_tuned.pkl")
SCALER_PATH        = os.path.join(TRUE_ROOT, "Membangun_model", "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(TRUE_ROOT, "preprocessing", "data", "output", "feature_names.pkl")
TRAIN_STATS_PATH   = os.path.join(TRUE_ROOT, "preprocessing", "data", "output", "X_train_scaled.npy")

print("\n" + "="*90)
print("PROMETHEUS EXPORTER - PATH CHECK (FINAL)")
print(f"Root Project     : {TRUE_ROOT}")
print(f"Model            : {MODEL_PATH} → exists: {os.path.exists(MODEL_PATH)}")
print(f"Scaler           : {SCALER_PATH} → exists: {os.path.exists(SCALER_PATH)}")
print(f"Feature Names    : {FEATURE_NAMES_PATH} → exists: {os.path.exists(FEATURE_NAMES_PATH)}")
print(f"Training Stats   : {TRAIN_STATS_PATH} → exists: {os.path.exists(TRAIN_STATS_PATH)}")
print("="*90 + "\n")

# ============================================================
# PROMETHEUS REGISTRY & METRICS
# ============================================================

registry = CollectorRegistry()

model_info = Info('wine_model_info', 'Information about the wine quality model', registry=registry)

model_accuracy = Gauge('model_accuracy_score', 'Model accuracy score', ['model_version'], registry=registry)
model_f1_score = Gauge('model_f1_score', 'Model F1 score', ['model_version'], registry=registry)
model_auc_score = Gauge('model_auc_roc_score', 'Model AUC-ROC score', ['model_version'], registry=registry)

prediction_distribution = Gauge('prediction_class_distribution', 'Distribution of predicted classes', 
                                ['class_label', 'model_version'], registry=registry)
prediction_confidence = Histogram('prediction_confidence_score', 'Prediction confidence distribution',
                                  buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0], registry=registry)

feature_mean = Gauge('input_feature_mean', 'Mean of input features (recent batch)', ['feature_name'], registry=registry)
feature_std  = Gauge('input_feature_std', 'Std of input features (recent batch)', ['feature_name'], registry=registry)
feature_drift_score = Gauge('feature_drift_score', 'Drift score vs training data', ['feature_name'], registry=registry)

data_quality_score = Gauge('input_data_quality_score', 'Data quality score (0-1)', registry=registry)
outlier_count = Counter('input_outliers_total', 'Total outliers detected', ['feature_name'], registry=registry)

cpu_usage_percent = Gauge('system_cpu_usage_percent', 'CPU usage %', registry=registry)
memory_usage_percent = Gauge('system_memory_usage_percent', 'Memory usage %', registry=registry)
disk_usage_percent = Gauge('system_disk_usage_percent', 'Disk usage %', ['mountpoint'], registry=registry)

model_load_time = Histogram('model_load_duration_seconds', 'Time to load model', registry=registry)

# ============================================================
# COLLECTOR CLASS
# ============================================================

class MetricsCollector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = os.getenv("MODEL_VERSION", "v1.0")
        self.training_stats = None
        self.prediction_buffer = []
        self.buffer_size = 1000

        self.load_model()
        self.load_training_stats()

    def load_model(self):
        start = time.time()
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                print("Model berhasil dimuat!")

            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)

            if os.path.exists(FEATURE_NAMES_PATH):
                with open(FEATURE_NAMES_PATH, 'rb') as f:
                    self.feature_names = pickle.load(f)
                print(f"Feature names loaded: {len(self.feature_names)} fitur")

            duration = time.time() - start
            model_load_time.observe(duration)

            if self.model:
                model_info.info({
                    'version': self.model_version,
                    'type': type(self.model).__name__,
                    'features': str(len(self.feature_names)) if self.feature_names else 'unknown',
                    'load_time_seconds': f"{duration:.3f}"
                })
        except Exception as e:
            print(f"Error loading model: {e}")

    def load_training_stats(self):
        if not os.path.exists(TRAIN_STATS_PATH):
            print("X_train_scaled.npy tidak ditemukan → drift detection dinonaktifkan")
            return
        try:
            X_train = np.load(TRAIN_STATS_PATH)
            self.training_stats = {
                'mean': X_train.mean(axis=0),
                'std': X_train.std(axis=0) + 1e-8
            }
            print("Training stats berhasil dimuat → DRIFT DETECTION AKTIF!")
        except Exception as e:
            print(f"Gagal load training stats: {e}")

    def update_system_metrics(self):
        cpu_usage_percent.set(psutil.cpu_percent(interval=None))
        memory_usage_percent.set(psutil.virtual_memory().percent)
        disk_usage_percent.labels(mountpoint="/").set(psutil.disk_usage('/').percent)

    def record_prediction(self, X, y_pred, probabilities):
        self.prediction_buffer.append(X)
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)

        if len(self.prediction_buffer) >= 10 and self.feature_names:
            buf = np.vstack(self.prediction_buffer)
            for i, name in enumerate(self.feature_names):
                feature_mean.labels(feature_name=name).set(float(buf[:, i].mean()))
                feature_std.labels(feature_name=name).set(float(buf[:, i].std()))

        unique, counts = np.unique(y_pred, return_counts=True)
        total = len(y_pred)
        for lbl, cnt in zip(unique, counts):
            prediction_distribution.labels(class_label=str(lbl), model_version=self.model_version).set(cnt / total)

        for prob in np.max(probabilities, axis=1):
            prediction_confidence.observe(float(prob))

        if self.training_stats and self.feature_names:
            current_mean = X.mean(axis=0)
            drift = np.abs(current_mean - self.training_stats['mean']) / self.training_stats['std']
            for i, name in enumerate(self.feature_names):
                feature_drift_score.labels(feature_name=name).set(float(drift[i]))

            z = np.abs((X - self.training_stats['mean']) / self.training_stats['std'])
            outliers = z > 3
            for i, name in enumerate(self.feature_names):
                if np.any(outliers[:, i]):
                    outlier_count.labels(feature_name=name).inc(np.sum(outliers[:, i]))

        data_quality_score.set(1.0 if not np.isnan(X).any() else 0.5)

collector = MetricsCollector()

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    collector.update_system_metrics()
    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": collector.model is not None,
        "features_loaded": collector.feature_names is not None,
        "drift_detection": collector.training_stats is not None,
        "model_version": collector.model_version,
        "timestamp": time.time()
    })

@app.route("/record_prediction", methods=["POST"])
def record_prediction():
    try:
        data = request.get_json()
        X = np.array(data["X"])
        y_pred = np.array(data["y_pred"])
        probas = np.array(data["probabilities"])
        collector.record_prediction(X, y_pred, probas)
        return jsonify({"status": "recorded"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update_model_metrics", methods=["POST"])
def update_model_metrics():
    data = request.get_json() or {}
    v = collector.model_version
    if "accuracy" in data: model_accuracy.labels(model_version=v).set(float(data["accuracy"]))
    if "f1_score" in data: model_f1_score.labels(model_version=v).set(float(data["f1_score"]))
    if "auc_score" in data: model_auc_score.labels(model_version=v).set(float(data["auc_score"]))
    return jsonify({"status": "success"}), 200

# Background updater
def background_updater():
    while True:
        time.sleep(15)
        collector.update_system_metrics()

threading.Thread(target=background_updater, daemon=True).start()

if __name__ == "__main__":
    port = 8000
    print(f"\nPROMETHEUS EXPORTER BERJALAN DI http://localhost:{port}/metrics")
    print("Health → http://localhost:8000/health")
    print("Record → POST http://localhost:8000/record_prediction")
    print("-" * 80)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)