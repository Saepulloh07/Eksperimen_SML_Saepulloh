import os
import sys
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Utility
# ============================================================
def safe_load_npy(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    return np.load(path)

def safe_load_pickle(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)

def try_set_tracking_uri_dagshub(username, repo, token):
    if not username or not repo:
        raise ValueError("DAGSHUB_USERNAME or DAGSHUB_REPO not provided")

    uri = f"https://dagshub.com/{username}/{repo}.mlflow"
    mlflow.set_tracking_uri(uri)

    client = MlflowClient()
    _ = client.list_experiments()
    return uri


# ============================================================
# 1. MLflow Setup
# ============================================================
print("=" * 60)
print("MLFLOW SETUP")
print("=" * 60)

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if DAGSHUB_USERNAME and DAGSHUB_REPO:
    try:
        print("→ Attempting DagsHub connection...")
        tracking_uri = try_set_tracking_uri_dagshub(DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN)
        print(f"✓ Connected to DagsHub: {tracking_uri}")
    except Exception as e:
        print(f"[WARN] DagsHub failed: {e}")
        print("→ Falling back to local MLflow")
        mlflow.set_tracking_uri("file:./mlruns")
else:
    print("→ No DagsHub credentials. Using local MLflow ...")
    mlflow.set_tracking_uri("file:./mlruns")

tracking_uri = mlflow.get_tracking_uri()
print(f"✓ MLflow tracking URI: {tracking_uri}")

experiment_name = "wine_quality_classification"
mlflow.set_experiment(experiment_name)
print(f"✓ Experiment set: {experiment_name}")


# ============================================================
# 2. Load Preprocessed Data
# ============================================================
print("\n" + "=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

base_preproc = os.path.join("..", "preprosessing", "data", "output")

X_train = safe_load_npy(os.path.join(base_preproc, "X_train_scaled.npy"))
X_test = safe_load_npy(os.path.join(base_preproc, "X_test_scaled.npy"))
y_train = safe_load_npy(os.path.join(base_preproc, "y_train.npy"))
y_test = safe_load_npy(os.path.join(base_preproc, "y_test.npy"))
feature_names = safe_load_pickle(os.path.join(base_preproc, "feature_names.pkl"))
scaler = joblib.load(os.path.join(base_preproc, "scaler.pkl"))

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set:     {X_test.shape}")
print(f"✓ Features:     {len(feature_names)}")

# Save scaler locally for logging artifact
local_scaler_path = "scaler.joblib"
joblib.dump(scaler, local_scaler_path)
print(f"✓ Local scaler saved: {local_scaler_path}")


# ============================================================
# 3. MLflow Training
# ============================================================
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1
}

with mlflow.start_run(run_name="random_forest_baseline") as run:
    print(f"\nRUN ID: {run.info.run_id}")

    # --------------------------
    # Logging Params
    # --------------------------
    for k, v in params.items():
        mlflow.log_param(k, v)

    mlflow.log_param("train_samples", int(X_train.shape[0]))
    mlflow.log_param("test_samples", int(X_test.shape[0]))
    mlflow.log_param("n_features", int(X_train.shape[1]))

    # --------------------------
    # Train Model
    # --------------------------
    print("\nTraining model ...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print("✓ Model trained")

    # --------------------------
    # Predictions
    # --------------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # probability if available
    y_test_proba = None
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:, 1]

    # --------------------------
    # Metrics
    # --------------------------
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, average="binary")
    recall = recall_score(y_test, y_test_pred, average="binary")
    f1 = f1_score(y_test, y_test_pred, average="binary")

    roc_auc = None
    if y_test_proba is not None:
        roc_auc = roc_auc_score(y_test, y_test_proba)

    mcc = matthews_corrcoef(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    cm = confusion_matrix(y_test, y_test_pred)

    specificity = None
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc if roc_auc is not None else -1,
        "mcc": mcc,
        "kappa": kappa,
        "cv_mean_acc": cv_mean,
        "cv_std_acc": cv_std,
        "specificity": specificity if specificity is not None else -1,
        "overfitting": train_acc - test_acc
    }

    for name, val in metrics.items():
        mlflow.log_metric(name, float(val))

    # ============================================================
    # Artifacts
    # ============================================================
    os.makedirs("artifacts", exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Feature Importance
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi.head(10), x="importance", y="feature")
    fi_path = "artifacts/feature_importance.png"
    plt.savefig(fi_path)
    plt.close()
    mlflow.log_artifact(fi_path)

    # Report
    report = classification_report(y_test, y_test_pred)
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Save scaler
    mlflow.log_artifact(local_scaler_path)

    # ============================================================
    # Log Model
    # ============================================================
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
    )

    # Tags
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("task", "classification")
    mlflow.set_tag("dataset", "wine_quality")

print("\nAll done! 🚀")
