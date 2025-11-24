# ==========================================
# modelling.py – VERSI PALING STABIL UNTUK CI/CD
# ==========================================

import os
import pickle
import argparse
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

from dotenv import load_dotenv

load_dotenv()  

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")


# =============== ARGUMENT PARSER ===============
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="all",
                    help="Nama model atau 'all' untuk semua model")
args = parser.parse_args()


# =============== SETUP ===============
os.makedirs("csv_output", exist_ok=True)
os.makedirs("data", exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_Classification")

# Autolog sekali saja
mlflow.sklearn.autolog(log_input_examples=True,
                       log_model_signatures=True,
                       log_models=True,
                       silent=True)

print("MLflow siap – autolog aktif")


# =============== LOAD DATA ===============
with open("data/preprocessing_objects.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]
feature_names = data["feature_names"]

print(f"Data loaded → Train {X_train.shape}, Test {X_test.shape}")


# =============== DAFTAR MODEL ===============
models = {
    "Logistic_Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision_Tree":       DecisionTreeClassifier(random_state=42),
    "Random_Forest":       RandomForestClassifier(random_state=42, n_estimators=200),
    "Gradient_Boosting":   GradientBoostingClassifier(random_state=42, n_estimators=200),
    "SVM":                 SVC(random_state=42, probability=True),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes":         GaussianNB(),
}

# Jika hanya satu model yang diminta
if args.model_name != "all" and args.model_name in models:
    models = {args.model_name: models[args.model_name]}


# =============== TRAINING & LOGGING ===============
results = []

# TIDAK ADA start_run() MANUAL → mlflow run . sudah buat run utama
for name, model in models.items():
    print(f"\nTraining → {name}")

    # Fit & predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    # Log metrik (autolog sudah log model + params)
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", pre)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    if auc is not None:
        mlflow.log_metric("roc_auc", auc)

    results.append({
        "Model":      name,
        "Accuracy":  round(acc, 4),
        "Precision": round(pre, 4),
        "Recall":    round(rec, 4),
        "F1-Score":  round(f1, 4),
        "ROC-AUC":   round(auc, 4) if auc is not None else "N/A"
    })

    print(f"   Accuracy = {acc:.4f} | F1 = {f1:.4f}" + (f" | AUC = {auc:.4f}" if auc else ""))


# =============== SIMPAN HASIL & MODEL TERBAIK ===============
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\nRINGKASAN HASIL")
print(results_df.to_string(index=False))

# Simpan CSV
results_df.to_csv("csv_output/model_comparison_results.csv", index=False)
mlflow.log_artifact("csv_output/model_comparison_results.csv")

# Simpan model terbaik
best_name = results_df.iloc[0]["Model"]
best_acc  = results_df.iloc[0]["Accuracy"]
best_model = models[best_name]
best_model.fit(X_train, y_train)  # retrain sekali lagi

model_path = f"data/best_model_{best_name}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

# Log model terbaik
mlflow.sklearn.log_model(best_model, artifact_path="best_model")
mlflow.log_artifact(model_path)
mlflow.log_metric("best_accuracy", best_acc)

print(f"\nModel Terbaik: {best_name} (Accuracy = {best_acc})")
print(f"Model disimpan → {model_path}")
print("SELESAI – semua artefak sudah di-log ke MLflow!")