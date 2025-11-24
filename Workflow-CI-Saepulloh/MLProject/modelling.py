# ==========================================
# MODELLING PENYAKIT JANTUNG - FIX UNTUK GITHUB ACTIONS
# ==========================================

import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# ==========================================
# 0. PASTIKAN FOLDER OUTPUT ADA
# ==========================================
os.makedirs("csv_output", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

# ==========================================
# 1. SETUP MLFLOW (HANYA SEKALI!)
# ==========================================
print("\n" + "="*60)
print("SETUP MLFLOW TRACKING")
print("="*60)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_Classification")

# AKTIFKAN AUTOLOG HANYA SEKALI DI AWAL!
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable_for_unsupported_versions=True,
    silent=True  # agar tidak terlalu berisik di CI
)

print("✓ MLflow siap (autolog aktif sekali di awal)")

# ==========================================
# 2. LOAD DATA PREPROCESSING
# ==========================================
print("\n" + "="*60)
print("MEMUAT DATA PREPROCESSING")
print("="*60)

try:
    with open('data/preprocessing_objects.pkl', 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("File data/preprocessing_objects.pkl tidak ditemukan. Pastikan step preprocessing sudah dijalankan!")

X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']
feature_names = data['feature_names']

print(f"✓ Train shape : {X_train.shape}")
print(f"✓ Test shape  : {X_test.shape}")
print(f"✓ Fitur ({len(feature_names)}): {feature_names}")

# ==========================================
# 3. DEFINISI MODEL
# ==========================================
print("\n" + "="*60)
print("DEFINISI MODEL")
print("="*60)

models = {
    'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision_Tree'      : DecisionTreeClassifier(random_state=42),
    'Random_Forest'      : RandomForestClassifier(random_state=42, n_estimators=200),
    'Gradient_Boosting'  : GradientBoostingClassifier(random_state=42, n_estimators=200),
    'SVM'                : SVC(random_state=42, probability=True),
    'KNN'                : KNeighborsClassifier(n_neighbors=5),
    'Naive_Bayes'        : GaussianNB()
}

print(f"✓ Total model yang akan dilatih: {len(models)}")

# ==========================================
# 4. TRAINING & EVALUASI
# ==========================================
print("\n" + "="*60)
print("MULAI TRAINING SEMUA MODEL")
print("="*60)

results = []

with mlflow.start_run(run_name="All_Models_Comparison", nested=False):
    for model_name, model in models.items():
        print(f"\n---> Training: {model_name}")

        with mlflow.start_run(run_name=model_name, nested=True):
            # Fit model
            model.fit(X_train, y_train)

            # Prediksi
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Hitung metrik
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred)
            roc_auc   = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

            # Log metrik manual (autolog sudah menangani model & parameter)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)

            # Simpan ke list hasil
            results.append({
                'Model'     : model_name,
                'Accuracy'  : accuracy,
                'Precision' : precision,
                'Recall'    : recall,
                'F1-Score'  : f1,
                'ROC-AUC'   : roc_auc if roc_auc is not None else 'N/A'
            })

            print(f"     Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}" if roc_auc else f"     Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

# ==========================================
# 5. RINGKASAN & SIMPAN HASIL
# ==========================================
print("\n" + "="*60)
print("RINGKASAN HASIL")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

# Simpan CSV (untuk artifact di GitHub Actions)
results_df.to_csv("csv_output/model_comparison_results.csv", index=False)
print("\n✓ File CSV disimpan di csv_output/model_comparison_results.csv")

# ==========================================
# 6. TENTUKAN & SIMPAN MODEL TERBAIK
# ==========================================
best_row = results_df.iloc[0]
best_model_name = best_row['Model']
best_accuracy = best_row['Accuracy']

print(f"\nTROFI Model Terbaik: {best_model_name} (Accuracy = {best_accuracy:.4f})")

# Retrain sekali lagi (agar 100% sama dengan yang dilog MLflow)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Simpan dengan pickle
best_model_path = f"data/best_model_{best_model_name.replace(' ', '_')}.pkl"
with open(best_model_path, 'wb') as f:
    pickle.dump(best_model, f)

# Log model terbaik sebagai artifact utama
with mlflow.start_run(run_name="Best_Model_Final"):
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_artifact(best_model_path)

print(f"✓ Model terbaik disimpan di: {best_model_path}")
print(f"✓ Artefak model terbaik juga di-log ke MLflow")

# ==========================================
# SELESAI
# ==========================================
print("\n" + "="*60)
print("MODELLING SELESAI TANPA ERROR!")
print("="*60)
print("   Semua run tersimpan di folder ./mlruns")
print("   Hasil perbandingan: csv_output/model_comparison_results.csv")
print(f"   Best model ({best_model_name}) → {best_model_path}")