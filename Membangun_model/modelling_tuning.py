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
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# -------------------------
# Utility helpers
# -------------------------
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


def try_set_tracking_uri_dagshub(username, repo, token=None):
    """
    Coba set tracking ke DagsHub. Jika gagal, raise exception.
    """
    if not username or not repo:
        raise ValueError("DAGSHUB_USERNAME or DAGSHUB_REPO not provided")
    uri = f"https://dagshub.com/{username}/{repo}.mlflow"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    # quick check (may raise)
    _ = client.list_experiments()
    return uri


# -------------------------
# 1. SETUP MLFLOW
# -------------------------
print("=" * 60)
print("HYPERPARAMETER TUNING WITH MLFLOW")
print("=" * 60)

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

use_local = False
if DAGSHUB_USERNAME and DAGSHUB_REPO:
    try:
        print("→ Attempting to connect to DagsHub MLflow tracking server...")
        tracking_uri = try_set_tracking_uri_dagshub(DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN)
        print(f"✓ Connected to DagsHub tracking URI: {tracking_uri}")
    except Exception as e:
        print(f"[WARN] Could not use DagsHub tracking: {e}")
        print("→ Falling back to local file-based MLflow tracking (file:./mlruns)")
        mlflow.set_tracking_uri("file:./mlruns")
        use_local = True
else:
    print("→ No DagsHub credentials. Using local MLflow ...")
    mlflow.set_tracking_uri("file:./mlruns")
    use_local = True

print(f"✓ MLflow URI: {mlflow.get_tracking_uri()}")

experiment_name = "wine_quality_classification"
try:
    mlflow.set_experiment(experiment_name)
    print(f"✓ Experiment: {experiment_name}")
except Exception as e:
    print(f"[WARN] set_experiment failed: {e}")
    if not use_local:
        print("→ Retrying with local MLflow")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)
        print(f"✓ Experiment (local): {experiment_name}")
    else:
        print("[ERROR] Could not set experiment even on local mlruns.")
        sys.exit(1)


# -------------------------
# 2. LOAD DATA
# -------------------------
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

# Adjust base_preproc if your preprocessed files are in a different folder.
base_preproc = os.path.join("..", "preprosessing", "data", "output")
# fallback: current directory
if not os.path.isdir(base_preproc):
    base_preproc = "."

X_train_path = os.path.join(base_preproc, "X_train_scaled.npy")
X_test_path = os.path.join(base_preproc, "X_test_scaled.npy")
y_train_path = os.path.join(base_preproc, "y_train.npy")
y_test_path = os.path.join(base_preproc, "y_test.npy")
feature_names_path = os.path.join(base_preproc, "feature_names.pkl")

X_train = safe_load_npy(X_train_path)
X_test = safe_load_npy(X_test_path)
y_train = safe_load_npy(y_train_path)
y_test = safe_load_npy(y_test_path)
feature_names = safe_load_pickle(feature_names_path)

print("✓ Data loaded successfully")
print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")
print(f"  n_features: {len(feature_names)}")


# -------------------------
# 3. GRID SEARCH HYPERPARAMETER TUNING
# -------------------------
print("\n" + "=" * 60)
print("GRID SEARCH HYPERPARAMETER TUNING")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

total_combinations = (
    len(param_grid['n_estimators'])
    * len(param_grid['max_depth'])
    * len(param_grid['min_samples_split'])
    * len(param_grid['min_samples_leaf'])
    * len(param_grid['max_features'])
)
print(f"\nParameter grid: {param_grid}")
print(f"Total combinations: {total_combinations}")

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1,
    return_train_score=True
)

# Start parent run for tuning
with mlflow.start_run(run_name="hyperparameter_tuning_parent") as parent_run:
    print(f"\n🚀 Parent Run ID: {parent_run.info.run_id}")

    # Log tuning metadata
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("total_combinations", total_combinations)

    print("\n--- Fitting Grid Search (this may take a while...) ---")
    grid_search.fit(X_train, y_train)
    print("✓ Grid search completed")

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print("\n--- Best Parameters ---")
    for p, v in best_params.items():
        print(f"✓ {p}: {v}")
        mlflow.log_param(f"best_{p}", v)
    mlflow.log_metric("best_cv_score", best_score)
    print(f"✓ Best CV Score: {best_score:.4f}")

    # -------------------------
    # Log top 5 configurations (nested runs)
    # -------------------------
    print("\n--- Logging Top 5 Configurations ---")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')

    for idx, row in results_df.head(5).iterrows():
        rank = int(row.get('rank_test_score', idx + 1))
        with mlflow.start_run(run_name=f"config_rank_{rank}", nested=True) as cfg_run:
            params = row['params']
            for param, value in params.items():
                mlflow.log_param(param, value)
            mlflow.log_metric("mean_test_score", float(row.get('mean_test_score', np.nan)))
            mlflow.log_metric("std_test_score", float(row.get('std_test_score', np.nan)))
            mlflow.log_metric("mean_train_score", float(row.get('mean_train_score', np.nan)))
            mlflow.log_metric("rank", float(rank))
            print(f"✓ Logged configuration rank {rank}")

    # -------------------------
    # Evaluate best model
    # -------------------------
    print("\n" + "=" * 60)
    print("EVALUATING BEST MODEL")
    print("=" * 60)

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Robust probability extraction
    y_test_proba = None
    if hasattr(best_model, "predict_proba"):
        try:
            # for multiclass, take proba for positive class if binary
            proba = best_model.predict_proba(X_test)
            if proba.shape[1] == 2:
                y_test_proba = proba[:, 1]
            else:
                # multiclass: compute ROC-AUC using 'ovr' later if needed
                y_test_proba = None
        except Exception:
            y_test_proba = None
    elif hasattr(best_model, "decision_function"):
        try:
            y_test_proba = best_model.decision_function(X_test)
        except Exception:
            y_test_proba = None

    # Metrics with fallbacks for multiclass
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Use binary metrics if 2 classes else macro
    average_method = 'binary' if len(np.unique(y_test)) == 2 else 'macro'
    precision = precision_score(y_test, y_test_pred, average=average_method, zero_division=0)
    recall = recall_score(y_test, y_test_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average=average_method, zero_division=0)

    roc_auc = None
    try:
        if y_test_proba is not None and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_test_proba)
    except Exception:
        roc_auc = None

    mcc = matthews_corrcoef(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)

    # Specificity only if binary
    cm = confusion_matrix(y_test, y_test_pred)
    specificity = None
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    metrics = {
        'final_train_accuracy': float(train_accuracy),
        'final_test_accuracy': float(test_accuracy),
        'final_precision': float(precision),
        'final_recall': float(recall),
        'final_f1_score': float(f1),
        'final_roc_auc': float(roc_auc) if roc_auc is not None else None,
        'final_matthews_corrcoef': float(mcc),
        'final_cohen_kappa': float(kappa),
        'final_cv_mean': cv_mean,
        'final_cv_std': cv_std,
        'final_specificity': float(specificity) if specificity is not None else None,
        'final_overfitting': float(train_accuracy - test_accuracy)
    }

    print("\n--- Final Model Metrics ---")
    for name, val in metrics.items():
        if val is None:
            print(f"✓ {name}: None (skipped logging)")
        else:
            mlflow.log_metric(name, val)
            print(f"✓ {name}: {val:.6f}")

    # -------------------------
    # Visualizations (saved + logged)
    # -------------------------
    print("\n--- Creating Visualizations ---")
    os.makedirs("artifacts", exist_ok=True)

    # Confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False)
        plt.title('Confusion Matrix - Best Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join("artifacts", "best_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        mlflow.log_artifact(cm_path)
    except Exception as e:
        print(f"[WARN] Could not create/log confusion matrix: {e}")

    # Feature importance
    try:
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=fi_df.head(10), x='importance', y='feature')
        fi_path = os.path.join("artifacts", "best_feature_importance.png")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=300)
        plt.close()
        mlflow.log_artifact(fi_path)
    except Exception as e:
        print(f"[WARN] Could not create/log feature importance: {e}")

    # Hyperparameter analysis plots (simple, safe)
    try:
        results_df['param_n_estimators'] = results_df['params'].apply(lambda p: p.get('n_estimators'))
        results_df['param_max_depth'] = results_df['params'].apply(lambda p: str(p.get('max_depth')))
        n_est_results = results_df.groupby('param_n_estimators')['mean_test_score'].mean()
        depth_results = results_df.groupby('param_max_depth')['mean_test_score'].mean()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(list(n_est_results.index), n_est_results.values, marker='o', linewidth=2)
        plt.xlabel('n_estimators')
        plt.ylabel('Mean CV Score')
        plt.title('n_estimators vs Performance')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(range(len(depth_results)), depth_results.values, marker='s', linewidth=2)
        plt.xlabel('max_depth (categories)')
        plt.ylabel('Mean CV Score')
        plt.title('max_depth vs Performance')
        plt.xticks(range(len(depth_results)), depth_results.index, rotation=45)
        plt.grid(True, alpha=0.3)

        hyp_path = os.path.join("artifacts", "hyperparameter_analysis.png")
        plt.tight_layout()
        plt.savefig(hyp_path, dpi=300)
        plt.close()
        mlflow.log_artifact(hyp_path)
    except Exception as e:
        print(f"[WARN] Could not create/log hyperparameter analysis: {e}")

    print("✓ All visualizations created and logged (where possible)")

    # -------------------------
    # Log best model
    # -------------------------
    print("\n--- Logging Best Model ---")
    try:
        from mlflow.models.signature import infer_signature
        try:
            signature = infer_signature(X_train[:5], best_model.predict(X_train[:5]))
        except Exception:
            signature = None

        try:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                signature=signature,
                registered_model_name="wine_quality_rf_tuned_model"
            )
            print("✓ Model logged and registration attempted.")
        except Exception as e_reg:
            # fallback to log without registry name
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                signature=signature
            )
            print(f"✓ Model logged without registry (fallback). Reason: {e_reg}")

        # Save model locally with joblib
        local_best_path = "best_model.pkl"
        joblib.dump(best_model, local_best_path)
        mlflow.log_artifact(local_best_path)
        print(f"✓ Best model saved locally and logged: {local_best_path}")
    except Exception as e:
        print(f"[ERROR] Failed to log best model: {e}")

    # -------------------------
    # Tuning summary artifact
    # -------------------------
    try:
        comparison_summary = []
        comparison_summary.append("HYPERPARAMETER TUNING RESULTS\n=============================\n")
        comparison_summary.append(f"Tuning Method: GridSearchCV\nCV folds: 5\nTotal Configs: {len(results_df)}\n\nBEST PARAMETERS:\n")
        for param, value in best_params.items():
            comparison_summary.append(f"- {param}: {value}\n")
        comparison_summary.append("\nBEST MODEL PERFORMANCE:\n")
        comparison_summary.append(f"- Best CV Score: {best_score:.4f}\n")
        comparison_summary.append(f"- Test Accuracy: {test_accuracy:.4f}\n")
        comparison_summary.append(f"- Precision: {precision:.4f}\n")
        comparison_summary.append(f"- Recall: {recall:.4f}\n")
        comparison_summary.append(f"- F1-Score: {f1:.4f}\n")
        comparison_summary.append(f"- ROC-AUC: {roc_auc if roc_auc is not None else 'N/A'}\n")
        comparison_summary.append(f"- Matthews Corr: {mcc:.4f}\n")
        comparison_summary.append(f"- Cohen Kappa: {kappa:.4f}\n")

        summary_path = "tuning_summary.txt"
        with open(summary_path, "w") as f:
            f.writelines(comparison_summary)
        mlflow.log_artifact(summary_path)
        print(f"✓ Tuning summary logged: {summary_path}")
    except Exception as e:
        print(f"[WARN] Could not create/log tuning summary: {e}")

    # Tags
    try:
        mlflow.set_tag("tuning_complete", "true")
        mlflow.set_tag("best_model", "logged")
        mlflow.set_tag("author", "MLOps_Engineer")
    except Exception:
        pass

    # Final prints
    print("\n" + "=" * 60)
    print("✅ HYPERPARAMETER TUNING COMPLETED!")
    print("=" * 60)
    print(f"\n📊 View results at: {mlflow.get_tracking_uri()}")
    print(f"🔑 Parent Run ID: {parent_run.info.run_id}")
    print(f"\n💡 Best model saved as: best_model.pkl")
