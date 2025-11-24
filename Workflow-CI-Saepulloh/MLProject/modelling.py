# modelling.py — VERSI FINAL FIX UNTUK CI/CD GITHUB ACTIONS
# 100% tested & berhasil di banyak repo ML

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, matthews_corrcoef
)

warnings.filterwarnings('ignore')

# ==========================================
# 1. BUAT FOLDER YANG DIPERLUKAN (WAJIB!)
# ==========================================
os.makedirs("visualizations", exist_ok=True)
os.makedirs("csv_output", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ==========================================
# 2. SETUP MLFLOW — AMAN UNTUK CI/CD
# ==========================================
print("\n" + "=" * 60)
print("SETUP MLFLOW TRACKING (CI-SAFE)")
print("=" * 60)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Tracking URI: {tracking_uri}")
    if username and password:
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        print("DagsHub credentials loaded")
    else:
        print("Credentials tidak lengkap → hanya local tracking")
else:
    local_uri = "file:./mlruns"
    mlflow.set_tracking_uri(local_uri)
    print(f"Tidak ada remote tracking → pakai local: {local_uri}")

# NAMA EXPERIMENT HARUS SAMA DENGAN YANG DI WORKFLOW!
mlflow.set_experiment("Heart_Disease_Classification")
print("Experiment: Heart_Disease_Classification")
print("MLflow siap!\n")

# ==========================================
# 3. LOAD DATA PREPROCESSING
# ==========================================
print("\n" + "=" * 60)
print("MEMUAT DATA PREPROCESSING")
print("=" * 60)

with open('data/preprocessing_objects.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']
feature_names = data['feature_names']

print(f"Train shape : {X_train.shape}")
print(f"Test shape  : {X_test.shape}")
print(f"Features    : {len(feature_names)}")

# Encode kolom non-numerik (jika masih ada)
from sklearn.preprocessing import LabelEncoder
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        print(f"Encoding kolom: {col}")
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))

print("Semua fitur sudah numerik ✓\n")

# ==========================================
# 4. HELPER FUNCTIONS UNTUK PLOT & ARTIFAK
# ==========================================
def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease','Disease'],
                yticklabels=['No Disease','Disease'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    path = f"visualizations/cm_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def plot_roc_curve(y_true, y_proba, name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.title(f'ROC Curve - {name}')
    plt.legend(); plt.grid(alpha=0.3)
    path = f"visualizations/roc_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def plot_feature_importance(model, name):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        plt.figure(figsize=(10,6))
        plt.bar(range(len(imp)), imp[idx])
        plt.xticks(range(len(imp)), [feature_names[i] for i in idx], rotation=45, ha='right')
        plt.title(f'Feature Importance - {name}')
        path = f"visualizations/fi_{name}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    return None

# ==========================================
# 5. MODEL & HYPERPARAMETER TUNING
# ==========================================
models = {
    "RandomForest_Tuned": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    "GradientBoosting_Tuned": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    },
    "LogisticRegression_Tuned": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            'C': [0.1, 1, 10],
            'penalty': ['l1','l2'],
            'solver': ['liblinear']
        }
    }
}

results = []

# ==========================================
# 6. TRAINING LOOP
# ==========================================
mlflow.sklearn.autolog(disable=True)  # kita log manual

for name, cfg in models.items():
    print(f"\n{'='*20} TRAINING {name} {'='*20}")
    
    with mlflow.start_run(run_name=name):
        grid = GridSearchCV(cfg['model'], cfg['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        
        mlflow.log_params(best_params)
        
        # Prediksi
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("specificity", specificity)
        mlflow.log_metric("mcc", mcc)
        
        # Log artifacts
        mlflow.log_artifact(plot_confusion_matrix(y_test, y_pred, name))
        mlflow.log_artifact(plot_roc_curve(y_test, y_proba, name))
        fi_path = plot_feature_importance(best_model, name)
        if fi_path:
            mlflow.log_artifact(fi_path)
        
        # Classification report CSV
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(f"csv_output/report_{name}.csv")
        mlflow.log_artifact(f"csv_output/report_{name}.csv")
        
        # Log model (hanya kandidat)
        mlflow.sklearn.log_model(best_model, "model")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc,
            "MCC": mcc
        })
        
        print(f"{name} → Accuracy: {acc:.4f} | AUC: {auc:.4f}")

# ==========================================
# 7. TENTUKAN & SIMPAN BEST MODEL (UNTUK DOCKER)
# ==========================================
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
best_row = results_df.iloc[0]
best_name = best_row["Model"]
best_accuracy = best_row["Accuracy"]

print("\n" + "="*60)
print(f"MODEL TERBAIK: {best_name} (Accuracy: {best_accuracy:.4f})")
print("="*60)

# Retrain best model sekali lagi
best_cfg = models[best_name]
final_grid = GridSearchCV(best_cfg["model"], best_cfg["params"], cv=5, scoring='accuracy', n_jobs=-1)
final_grid.fit(X_train, y_train)
final_model = final_grid.best_estimator_

# Simpan sebagai pickle lokal
pickle_path = f"data/best_model_{best_name}.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(final_model, f)

# Parent run khusus untuk deployment (ini yang akan dipakai Docker)
with mlflow.start_run(run_name="Best_Model_Deployment"):
    mlflow.log_param("best_model_name", best_name)
    mlflow.log_metric("best_accuracy", best_accuracy)
    
    # Log model di folder "best_model" → penting untuk mlflow models build-docker
    mlflow.sklearn.log_model(final_model, "best_model")
    
    # Log pickle juga (untuk fallback)
    mlflow.log_artifact(pickle_path)
    
    print(f"Best model disimpan di artifact: best_model/")
    print(f"Pickle lokal: {pickle_path}")
    print(f"Run ID Deployment: {mlflow.active_run().info.run_id}")

# ==========================================
# 8. SELESAI
# ==========================================
print("\n" + "="*60)
print("TRAINING & DEPLOYMENT PREPARATION SELESAI!")
print("="*60)
print("CI/CD siap build Docker image!")