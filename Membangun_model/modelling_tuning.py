# Konfigurasi DagsHub 
import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP MLFLOW + DAGSHUB
# ==========================================
print("\n" + "=" * 60)
print("SETUP MLFLOW TRACKING (DAGSHUB)")
print("=" * 60)

# Load environment variables dari file .env
load_dotenv()

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")


# Set tracking URI ke DagsHub
dagshub_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
mlflow.set_tracking_uri(dagshub_uri)

# Set credentials (jika repo private)
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

# Set experiment
mlflow.set_experiment("Heart_Disease_Tuned_Models")

print(f"✓ MLflow Tracking URI: {dagshub_uri}")
print("✓ Experiment: Heart_Disease_Tuned_Models")
print("\n💡 Dashboard akan tersedia di:")
print(f"   https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments")

# ==========================================
# 2. LOAD DATA PREPROCESSING
# ==========================================
print("\n" + "=" * 60)
print("MEMUAT DATA PREPROCESSING")
print("=" * 60)

with open('data/preprocessing_objects.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

print(f"✓ Data train: {X_train.shape}")
print(f"✓ Data test: {X_test.shape}")
print(f"✓ Jumlah fitur: {len(feature_names)}")

# Cek dan encode kolom non-numerik
print("\n--- MENGECEK TIPE DATA ---")
print(X_train.dtypes)

# Encode kolom 'dataset' atau 'asal_studi' jika ada dan masih object/string
from sklearn.preprocessing import LabelEncoder
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'string':
        print(f"⚠ Kolom '{col}' masih bertipe {X_train[col].dtype}, akan di-encode...")
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        print(f"✓ Kolom '{col}' berhasil di-encode")

print(f"\n✓ Semua fitur sudah numerik")

# ==========================================
# 3. FUNGSI HELPER UNTUK ARTIFACTS
# ==========================================

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Membuat plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    # Simpan plot
    plot_path = f'visualizations/cm_{model_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_roc_curve_plot(y_true, y_pred_proba, model_name):
    """Membuat plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = f'visualizations/roc_{model_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_feature_importance_plot(model, feature_names, model_name):
    """Membuat plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in indices], 
                   rotation=45, ha='right')
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        plot_path = f'visualizations/feature_imp_{model_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    return None

def create_precision_recall_curve_plot(y_true, y_pred_proba, model_name):
    """Membuat plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = f'visualizations/pr_curve_{model_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

# ==========================================
# 4. DEFINISI MODEL + HYPERPARAMETER TUNING
# ==========================================
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# Definisi model dan parameter grid
models_tuning = {
    'Random_Forest_Tuned': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Gradient_Boosting_Tuned': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'Logistic_Regression_Tuned': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    }
}

results = []

# ==========================================
# 5. TRAINING & MANUAL LOGGING
# ==========================================

for model_name, config in models_tuning.items():
    print(f"\n{'='*60}")
    print(f"Tuning Model: {model_name}")
    print(f"{'='*60}")
    
    # NONAKTIFKAN autolog untuk manual logging
    mlflow.sklearn.autolog(disable=True)
    
    with mlflow.start_run(run_name=model_name):
        
        # Grid Search
        print("⏳ Melakukan Grid Search...")
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"✓ Best Parameters: {best_params}")
        
        # Log best parameters
        mlflow.log_params(best_params)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"✓ CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Prediksi
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Evaluasi metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        
        # === MANUAL LOGGING METRICS ===
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_mean_accuracy", cv_mean)
        mlflow.log_metric("cv_std_accuracy", cv_std)
        
        # ARTEFAK TAMBAHAN #1: Specificity & NPV
        mlflow.log_metric("specificity", specificity)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        mlflow.log_metric("npv", npv)  # Negative Predictive Value
        
        # ARTEFAK TAMBAHAN #2: Matthews Correlation Coefficient
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(y_test, y_pred)
        mlflow.log_metric("matthews_corrcoef", mcc)
        
        # Log confusion matrix values
        mlflow.log_metric("true_negatives", int(tn))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("true_positives", int(tp))
        
        # === MANUAL LOGGING ARTIFACTS ===
        
        # 1. Confusion Matrix Plot
        cm_plot_path = create_confusion_matrix_plot(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_plot_path)
        
        # 2. ROC Curve Plot
        roc_plot_path = create_roc_curve_plot(y_test, y_pred_proba, model_name)
        mlflow.log_artifact(roc_plot_path)
        
        # 3. Feature Importance Plot (jika ada)
        fi_plot_path = create_feature_importance_plot(best_model, feature_names, model_name)
        if fi_plot_path:
            mlflow.log_artifact(fi_plot_path)
        
        # 4. Precision-Recall Curve (ARTEFAK TAMBAHAN #3)
        pr_plot_path = create_precision_recall_curve_plot(y_test, y_pred_proba, model_name)
        mlflow.log_artifact(pr_plot_path)
        
        # 5. Classification Report (ARTEFAK TAMBAHAN #4)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = f'csv_output/classification_report_{model_name}.csv'
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)
        
        # 6. Feature Importance CSV (ARTEFAK TAMBAHAN #5)
        if hasattr(best_model, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            fi_path = f'csv_output/feature_importance_{model_name}.csv'
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Tampilkan hasil
        print(f"\n📊 HASIL EVALUASI:")
        print(f"   Accuracy          : {accuracy:.4f}")
        print(f"   Precision         : {precision:.4f}")
        print(f"   Recall            : {recall:.4f}")
        print(f"   F1-Score          : {f1:.4f}")
        print(f"   ROC-AUC           : {roc_auc:.4f}")
        print(f"   Specificity       : {specificity:.4f}")
        print(f"   NPV               : {npv:.4f}")
        print(f"   Matthews Corr Coef: {mcc:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {cm}")
        
        # Simpan hasil
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Specificity': specificity,
            'MCC': mcc,
            'CV_Mean': cv_mean
        })
        
        print(f"✓ Model {model_name} berhasil dilog ke DagsHub")

# ==========================================
# 6. RINGKASAN HASIL
# ==========================================
print("\n" + "=" * 60)
print("RINGKASAN HASIL SEMUA MODEL")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

# Simpan hasil ke CSV
results_df.to_csv('csv_output/model_tuning_comparison.csv', index=False)
print("\n✓ Hasil perbandingan disimpan: csv_output/model_tuning_comparison.csv")

# ==========================================
# 7. MODEL TERBAIK
# ==========================================
print("\n" + "=" * 60)
print("MODEL TERBAIK")
print("=" * 60)

best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n🏆 Model Terbaik: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")

# ==========================================
# SELESAI
# ==========================================
print("\n" + "=" * 60)
print("MODELLING TUNING SELESAI!")
print("=" * 60)
print("\n Untuk melihat hasil di DagsHub:")
print(f"   https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments")
