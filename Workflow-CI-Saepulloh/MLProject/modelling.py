# ==========================================
# modelling.py – VERSI AMAN UNTUK GITHUB ACTIONS + mlflow run .
# ==========================================

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# 0. Buat folder yang diperlukan
# ==========================================
os.makedirs("csv_output", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ==========================================
# 1. Setup MLflow (hanya sekali!)
# ==========================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_Classification")

# Autolog SKLEARN AKTIF HANYA SEKALI
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=True
)

print("MLflow siap – autolog aktif")

# ==========================================
# 2. Load data preprocessing
# ==========================================
with open('data/preprocessing_objects.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
feature_names = data['feature_names']

print(f"Data loaded – Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 3. Daftar model
# ==========================================
models = {
    "Logistic_Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision_Tree":       DecisionTreeClassifier(random_state=42),
    "Random_Forest":       RandomForestClassifier(random_state=42, n_estimators=200),
    "Gradient_Boosting":   GradientBoostingClassifier(random_state=42, n_estimators=200),
    "SVM":                 SVC(random_state=42, probability=True),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes":       GaussianNB(),
}

# ==========================================
# 4. Training + evaluasi (PAKAI NESTED RUN SAJA!)
# ==========================================
results = []

# TIDAK PERLU start_run() di sini → mlflow run . sudah buatkan run utama
for model_name, model in models.items():
    print(f"\nTraining → {model_name}")

    # Nested run untuk tiap model (ini yang benar!)
    with mlflow.start_run(run_name=model_name, nested=True):
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        # Log manual (autolog sudah log model + params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", pre)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        if auc is not None:
            mlflow.log_metric("roc_auc", auc)

        results.append({
            "Model":      model_name,
            "Accuracy":  round(acc, 4),
            "Precision": round(pre, 4),
            "Recall":    round(rec, 4),
            "F1-Score":  round(f1, 4),
            "ROC-AUC":   round(auc, 4) if auc is not None else "N/A"
        })

        print(f"   Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}" if auc else f"   Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ==========================================
# 5. Simpan ringkasan
# ==========================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

print("\nRINGKASAN HASIL")
print(results_df.to_string(index=False))

results_df.to_csv("csv_output/model_comparison_results.csv", index=False)

# ==========================================
# 6. Simpan model terbaik
# ==========================================
best_model_name = results_df.iloc[0]["Model"]
best_accuracy   = results_df.iloc[0]["Accuracy"]

print(f"\nModel Terbaik: {best_model_name} (Accuracy = {best_accuracy})")

# Retrain & simpan
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

model_path = f"data/best_model_{best_model_name}.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

# Log model terbaik sebagai artifact di run utama (bukan nested)
mlflow.log_artifact(model_path, artifact_path="best_model")
mlflow.log_metric("best_accuracy", best_accuracy)

print(f"Model terbaik disimpan → {model_path}")
print("SELESAI! Semua run tersimpan di folder ./mlruns")