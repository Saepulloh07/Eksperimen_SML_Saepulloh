import os
import sys
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, matthews_corrcoef,
    cohen_kappa_score, log_loss, balanced_accuracy_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import config
from mlflow_config import setup_mlflow, get_or_create_experiment


def load_data():
    """Load preprocessed data"""
    print("=" * 70)
    print("📥 LOADING PREPROCESSED DATA")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_path = os.path.join(script_dir, "..", "preprocessing", "data", "output")
    base_path = os.path.normpath(base_path)

    files = {
        'X_train': 'X_train_scaled.npy',
        'X_test': 'X_test_scaled.npy',
        'y_train': 'y_train.npy',
        'y_test': 'y_test.npy',
        'feature_names': 'feature_names.pkl',
        'scaler': 'scaler.pkl'
    }
    
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ File not found: {filepath}")
        
        if filename.endswith('.npy'):
            data[key] = np.load(filepath)
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data[key] = pickle.load(f)
    
    print(f"✅ Data loaded successfully from: {base_path}")
    print(f"   - X_train: {data['X_train'].shape}")
    print(f"   - X_test: {data['X_test'].shape}")
    print(f"   - Features: {len(data['feature_names'])}")
    
    return data


def calculate_metrics(model, X_train, X_test, y_train, y_test):
    """
    Calculate comprehensive metrics
    """
    print("\n📊 Calculating metrics...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    metrics = {
        # Accuracy metrics
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        
        # Classification metrics
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        
        # ROC & Probability metrics
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'log_loss': log_loss(y_test, y_test_proba),
        
        # Statistical metrics
        'matthews_corrcoef': matthews_corrcoef(y_test, y_test_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_test_pred),
        
        # Overfitting indicator
        'overfitting_gap': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred)
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    # Confusion matrix specificity
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['specificity'] = tn / (tn + fp)
    metrics['sensitivity'] = tp / (tp + fn)  # same as recall
    
    # False positive/negative rates
    metrics['false_positive_rate'] = fp / (fp + tn)
    metrics['false_negative_rate'] = fn / (fn + tp)
    
    print("✅ Metrics calculated:")
    for name, value in metrics.items():
        print(f"   - {name}: {value:.4f}")
    
    return metrics, y_test_pred, y_test_proba


def create_visualizations(model, X_test, y_test, y_pred, feature_names):
    """
    Create and save visualizations
    """
    print("\n🎨 Creating visualizations...")
    
    os.makedirs("artifacts", exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bad Wine', 'Good Wine'],
                yticklabels=['Bad Wine', 'Good Wine'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 2. Feature Importance
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_df.head(10), x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=300)
    plt.close()
    
    # 3. Classification Report
    report = classification_report(y_test, y_pred, 
                                   target_names=['Bad Wine', 'Good Wine'])
    with open('artifacts/classification_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Visualizations saved to artifacts/")
    
    return ['artifacts/confusion_matrix.png', 
            'artifacts/feature_importance.png',
            'artifacts/classification_report.txt']


def train_model():
    """
    Main training function with MLflow tracking
    """
    # Setup MLflow
    print("=" * 70)
    print("🚀 WINE QUALITY MODEL TRAINING")
    print("=" * 70)
    
    setup_mlflow()
    get_or_create_experiment("wine_quality_classification")
    
    # Load data
    data = load_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    # Model parameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_baseline") as run:
        print(f"\n📌 Run ID: {run.info.run_id}")
        print(f"📌 Run Name: {run.info.run_name}")
        
        # Log parameters
        print("\n📝 Logging parameters...")
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Additional metadata
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("model_type", "RandomForest")
        
        # Train model
        print("\n🏋️  Training model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        print("✅ Model trained successfully")
        
        # Calculate metrics
        metrics, y_pred, y_proba = calculate_metrics(
            model, X_train, X_test, y_train, y_test
        )
        
        # Log metrics
        print("\n📊 Logging metrics...")
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))
        
        # Create visualizations
        artifact_paths = create_visualizations(
            model, X_test, y_test, y_pred, feature_names
        )
        
        # Log artifacts
        print("\n📦 Logging artifacts...")
        for path in artifact_paths:
            mlflow.log_artifact(path)
        
        # Log model
        print("\n💾 Logging model...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="wine_quality_rf_model"
        )
        
        # Save model locally
        joblib.dump(model, 'wine_quality_model.pkl')
        mlflow.log_artifact('wine_quality_model.pkl')
        
        # Log scaler
        mlflow.log_artifact('../preprocessing/data/output/scaler.pkl')
        
        # Set tags
        mlflow.set_tag("developer", "Saepulloh")
        mlflow.set_tag("algorithm", "RandomForest")
        mlflow.set_tag("dataset", "Wine Quality")
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📊 Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
        print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"\n🌐 View results: {mlflow.get_tracking_uri()}")
        print(f"🔑 Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train_model()