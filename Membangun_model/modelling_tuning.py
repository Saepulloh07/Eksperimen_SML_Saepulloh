import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef, cohen_kappa_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===== 1. SETUP MLFLOW =====
print("=" * 60)
print("HYPERPARAMETER TUNING WITH MLFLOW")
print("=" * 60)

# Set tracking URI (sesuaikan dengan setup Anda)
DAGSHUB_USERNAME = "your_username"
DAGSHUB_REPO = "wine-quality-mlops"

mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow')
# Atau local: mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "wine_quality_classification"
mlflow.set_experiment(experiment_name)

print(f"✓ MLflow URI: {mlflow.get_tracking_uri()}")
print(f"✓ Experiment: {experiment_name}")

# ===== 2. LOAD DATA =====
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"✓ Data loaded successfully")

# ===== 3. HYPERPARAMETER TUNING =====
print("\n" + "=" * 60)
print("GRID SEARCH HYPERPARAMETER TUNING")
print("=" * 60)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print(f"\nParameter grid: {param_grid}")
print(f"Total combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])}")

# Create base model
base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Setup GridSearchCV
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
    
    # Fit grid search
    print("\n--- Fitting Grid Search (This may take a while...) ---")
    grid_search.fit(X_train, y_train)
    print("✓ Grid search completed")
    
    # Get results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    print(f"\n--- Best Parameters ---")
    for param, value in best_params.items():
        print(f"✓ {param}: {value}")
    print(f"\n✓ Best CV Score: {best_score:.4f}")
    
    # Log parent run parameters
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("total_combinations", len(grid_search.cv_results_['params']))
    
    # Log best parameters
    for param, value in best_params.items():
        mlflow.log_param(f"best_{param}", value)
    
    mlflow.log_metric("best_cv_score", best_score)
    
    # ===== 4. LOG TOP 5 CONFIGURATIONS =====
    print("\n--- Logging Top 5 Configurations ---")
    
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    for idx, row in results_df.head(5).iterrows():
        with mlflow.start_run(run_name=f"config_rank_{row['rank_test_score']}", nested=True):
            
            # Log parameters
            params = row['params']
            for param, value in params.items():
                mlflow.log_param(param, value)
            
            # Log metrics
            mlflow.log_metric("mean_test_score", row['mean_test_score'])
            mlflow.log_metric("std_test_score", row['std_test_score'])
            mlflow.log_metric("mean_train_score", row['mean_train_score'])
            mlflow.log_metric("rank", row['rank_test_score'])
            
            print(f"✓ Logged configuration rank {row['rank_test_score']}")
    
    # ===== 5. EVALUATE BEST MODEL =====
    print("\n" + "=" * 60)
    print("EVALUATING BEST MODEL")
    print("=" * 60)
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate all metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Log all metrics
    metrics = {
        'final_train_accuracy': train_accuracy,
        'final_test_accuracy': test_accuracy,
        'final_precision': precision,
        'final_recall': recall,
        'final_f1_score': f1,
        'final_roc_auc': roc_auc,
        'final_matthews_corrcoef': mcc,
        'final_cohen_kappa': kappa,
        'final_cv_mean': cv_mean,
        'final_cv_std': cv_std,
        'final_specificity': specificity,
        'final_overfitting': train_accuracy - test_accuracy
    }
    
    print("\n--- Final Model Metrics ---")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
        print(f"✓ {metric_name}: {metric_value:.4f}")
    
    # ===== 6. CREATE VISUALIZATIONS =====
    print("\n--- Creating Visualizations ---")
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False)
    plt.title('Confusion Matrix - Best Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('best_confusion_matrix.png', dpi=300)
    mlflow.log_artifact('best_confusion_matrix.png')
    plt.close()
    
    # Feature Importance
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='magma')
    plt.title('Top 10 Feature Importance - Best Model', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('best_feature_importance.png', dpi=300)
    mlflow.log_artifact('best_feature_importance.png')
    plt.close()
    
    # Hyperparameter importance plot
    plt.figure(figsize=(12, 6))
    
    # Plot for n_estimators
    plt.subplot(1, 2, 1)
    n_est_results = results_df.groupby('param_n_estimators')['mean_test_score'].mean()
    plt.plot(n_est_results.index, n_est_results.values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('n_estimators', fontsize=12)
    plt.ylabel('Mean CV Score', fontsize=12)
    plt.title('n_estimators vs Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot for max_depth
    plt.subplot(1, 2, 2)
    depth_results = results_df.groupby('param_max_depth')['mean_test_score'].mean()
    plt.plot(range(len(depth_results)), depth_results.values, marker='s', linewidth=2, markersize=8, color='coral')
    plt.xlabel('max_depth', fontsize=12)
    plt.ylabel('Mean CV Score', fontsize=12)
    plt.title('max_depth vs Performance', fontsize=14, fontweight='bold')
    plt.xticks(range(len(depth_results)), depth_results.index)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png', dpi=300)
    mlflow.log_artifact('hyperparameter_analysis.png')
    plt.close()
    
    print("✓ All visualizations created and logged")
    
    # ===== 7. LOG BEST MODEL =====
    print("\n--- Logging Best Model ---")
    
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, best_model.predict(X_train))
    
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        signature=signature,
        registered_model_name="wine_quality_rf_tuned_model"
    )
    
    # Save model locally
    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    mlflow.log_artifact('best_model.pkl')
    
    print("✓ Best model logged successfully")
    
    # ===== 8. COMPARISON SUMMARY =====
    comparison_summary = f"""
    HYPERPARAMETER TUNING RESULTS
    =============================
    
    Tuning Method: GridSearchCV
    Cross-Validation Folds: 5
    Total Configurations Tested: {len(grid_search.cv_results_['params'])}
    
    BEST PARAMETERS:
    {'-'*40}
    """
    for param, value in best_params.items():
        comparison_summary += f"{param}: {value}\n    "
    
    comparison_summary += f"""
    
    BEST MODEL PERFORMANCE:
    {'-'*40}
    Best CV Score: {best_score:.4f}
    Test Accuracy: {test_accuracy:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1-Score: {f1:.4f}
    ROC-AUC: {roc_auc:.4f}
    Matthews Correlation: {mcc:.4f}
    Cohen's Kappa: {kappa:.4f}
    
    IMPROVEMENT ANALYSIS:
    {'-'*40}
    (Compare these results with baseline model)
    """
    
    with open('tuning_summary.txt', 'w') as f:
        f.write(comparison_summary)
    
    mlflow.log_artifact('tuning_summary.txt')
    
    # Add tags
    mlflow.set_tag("tuning_complete", "true")
    mlflow.set_tag("best_model", "logged")
    mlflow.set_tag("author", "MLOps_Engineer")
    
    print("\n" + "=" * 60)
    print("✅ HYPERPARAMETER TUNING COMPLETED!")
    print("=" * 60)
    print(f"\n📊 View results at: {mlflow.get_tracking_uri()}")
    print(f"🔑 Parent Run ID: {parent_run.info.run_id}")
    print(f"\n💡 Best model saved as: best_model.pkl")