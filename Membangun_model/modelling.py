import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, matthews_corrcoef, cohen_kappa_score)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===== 1. SETUP MLFLOW =====
print("=" * 60)
print("MLFLOW SETUP")
print("=" * 60)

# Set tracking URI ke DagsHub (ganti dengan credentials Anda)
# Format: https://dagshub.com/USERNAME/REPO_NAME.mlflow
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Setup DagsHub tracking
mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow')

# Untuk local tracking (comment jika sudah setup DagsHub):
# mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
experiment_name = "wine_quality_classification"
mlflow.set_experiment(experiment_name)

print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"✓ Experiment: {experiment_name}")

# ===== 2. LOAD PREPROCESSED DATA =====
print("\n" + "=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

X_train = np.load('../preprosessing/data/output/X_train_scaled.npy')
X_test = np.load('../preprosessing/data/output/X_test_scaled.npy')
y_train = np.load('../preprosessing/data/output/y_train.npy')
y_test = np.load('../preprosessing/data/output/y_test.npy')

with open('../preprosessing/data/output/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('../preprosessing/data/output/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")
print(f"✓ Features: {len(feature_names)}")

# ===== 3. MODEL TRAINING WITH MLFLOW =====
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Start MLflow run
with mlflow.start_run(run_name="random_forest_baseline") as run:
    
    print(f"\n MLflow Run ID: {run.info.run_id}")
    
    # ===== 3.1 Log Parameters =====
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("\n--- Logging Parameters ---")
    for param, value in params.items():
        mlflow.log_param(param, value)
        print(f"✓ {param}: {value}")
    
    # Log dataset info
    mlflow.log_param("dataset_name", "wine_quality")
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])
    mlflow.log_param("n_features", X_train.shape[1])
    
    # ===== 3.2 Train Model =====
    print("\n--- Training Model ---")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")
    
    # ===== 3.3 Make Predictions =====
    print("\n--- Making Predictions ---")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    print("✓ Predictions completed")
    
    # ===== 3.4 Calculate Metrics =====
    print("\n--- Calculating Metrics ---")
    
    # Standard metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # Additional metrics (manual logging)
    mcc = matthews_corrcoef(y_test, y_test_pred)  # Matthews Correlation Coefficient
    kappa = cohen_kappa_score(y_test, y_test_pred)  # Cohen's Kappa
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Log all metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'matthews_corrcoef': mcc,  # Additional metric 1
        'cohen_kappa': kappa,  # Additional metric 2
        'cv_mean_accuracy': cv_mean,  # Additional metric 3
        'cv_std_accuracy': cv_std,  # Additional metric 4
        'specificity': specificity,  # Additional metric 5
        'overfitting_score': train_accuracy - test_accuracy  # Additional metric 6
    }
    
    print("\n--- Logging Metrics ---")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
        print(f"✓ {metric_name}: {metric_value:.4f}")
    
    # ===== 3.5 Log Artifacts =====
    print("\n--- Creating & Logging Artifacts ---")
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('confusion_matrix.png')
    print("✓ confusion_matrix.png")
    plt.close()
    
    # Feature Importance
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importance', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('feature_importance.png')
    print("✓ feature_importance.png")
    plt.close()
    
    # Classification Report
    report = classification_report(y_test, y_test_pred, target_names=['Bad Wine', 'Good Wine'])
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt')
    print("✓ classification_report.txt")
    
    # Model Summary
    summary = f"""
    MODEL TRAINING SUMMARY
    =====================
    
    Model: Random Forest Classifier
    Dataset: Wine Quality (Binary Classification)
    
    Dataset Info:
    - Training samples: {X_train.shape[0]}
    - Test samples: {X_test.shape[0]}
    - Number of features: {X_train.shape[1]}
    
    Model Performance:
    - Train Accuracy: {train_accuracy:.4f}
    - Test Accuracy: {test_accuracy:.4f}
    - Precision: {precision:.4f}
    - Recall: {recall:.4f}
    - F1-Score: {f1:.4f}
    - ROC-AUC: {roc_auc:.4f}
    - Matthews Correlation Coefficient: {mcc:.4f}
    - Cohen's Kappa: {kappa:.4f}
    - CV Mean Accuracy: {cv_mean:.4f} (±{cv_std:.4f})
    - Specificity: {specificity:.4f}
    - Overfitting Score: {train_accuracy - test_accuracy:.4f}
    
    Confusion Matrix:
    {cm}
    """
    
    with open('model_summary.txt', 'w') as f:
        f.write(summary)
    mlflow.log_artifact('model_summary.txt')
    print("✓ model_summary.txt")
    
    # ===== 3.6 Log Model =====
    print("\n--- Logging Model ---")
    
    # Create signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Log model with signature
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="wine_quality_rf_model"
    )
    print("✓ Model logged successfully")
    
    # Save scaler as artifact
    mlflow.log_artifact('scaler.pkl')
    print("✓ Scaler logged")
    
    # ===== 3.7 Add Tags =====
    print("\n--- Adding Tags ---")
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("task", "binary_classification")
    mlflow.set_tag("dataset", "wine_quality")
    mlflow.set_tag("author", "MLOps_Engineer")
    mlflow.set_tag("version", "1.0.0")
    print("✓ Tags added")
    
