# ==========================================
# MODELLING PENYAKIT JANTUNG - MLFLOW AUTOLOG
# ==========================================

import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP MLFLOW
# ==========================================
print("\n" + "=" * 60)
print("SETUP MLFLOW TRACKING (LOCAL)")
print("=" * 60)

# Set tracking URI lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_Classification")

print("✓ MLflow Tracking URI: ./mlruns")
print("✓ Experiment: Heart_Disease_Classification")
print("\n💡 Untuk melihat dashboard:")
print("   mlflow ui --port 5000")
print("   Buka: http://localhost:5000")

# ==========================================
# 2. LOAD DATA PREPROCESSING
# ==========================================
print("\n" + "=" * 60)
print("MEMUAT DATA PREPROCESSING")
print("=" * 60)

# Load data dari PKL
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
print(f"✓ Fitur: {feature_names}")

# ==========================================
# 3. DEFINISI MODEL
# ==========================================
print("\n" + "=" * 60)
print("DEFINISI MODEL")
print("=" * 60)

models = {
    'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision_Tree': DecisionTreeClassifier(random_state=42),
    'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient_Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive_Bayes': GaussianNB()
}

print(f"✓ Total model: {len(models)}")
for name in models.keys():
    print(f"  • {name}")

# ==========================================
# 4. TRAINING & EVALUASI MODEL
# ==========================================
print("\n" + "=" * 60)
print("TRAINING & EVALUASI MODEL")
print("=" * 60)

results = []

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training Model: {model_name}")
    print(f"{'='*60}")
    
    # Aktifkan autolog MLflow
    mlflow.sklearn.autolog(log_input_examples=True, 
                           log_model_signatures=True,
                           log_models=True)
    
    with mlflow.start_run(run_name=model_name, nested=True):
        # Training model
        print("⏳ Training model...")
        model.fit(X_train, y_train)
        print("✓ Training selesai")
        
        # Prediksi
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        # Log tambahan metrics (manual)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        if roc_auc:
            mlflow.log_metric("roc_auc", roc_auc)
        
        # Tampilkan hasil
        print(f"\n📊 HASIL EVALUASI:")
        print(f"   Accuracy  : {accuracy:.4f}")
        print(f"   Precision : {precision:.4f}")
        print(f"   Recall    : {recall:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        if roc_auc:
            print(f"   ROC-AUC   : {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   {cm}")
        
        # Simpan hasil
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc if roc_auc else 'N/A'
        })
        
        print(f"✓ Model {model_name} berhasil dilog ke MLflow")

# ==========================================
# 5. RINGKASAN HASIL
# ==========================================
print("\n" + "=" * 60)
print("RINGKASAN HASIL SEMUA MODEL")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

# Simpan hasil ke CSV
results_df.to_csv('csv_output/model_comparison_results.csv', index=False)
print("\n✓ Hasil perbandingan disimpan: csv_output/model_comparison_results.csv")

# ==========================================
# 6. MODEL TERBAIK
# ==========================================
print("\n" + "=" * 60)
print("MODEL TERBAIK")
print("=" * 60)

best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n🏆 Model Terbaik: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")

# ==========================================
# 7. SAVE BEST MODEL
# ==========================================
print("\n" + "=" * 60)
print("MENYIMPAN MODEL TERBAIK")
print("=" * 60)

# Retrain best model dan simpan
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Simpan dengan pickle
model_path = f'data/best_model_{best_model_name}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✓ Model terbaik disimpan: {model_path}")

# ==========================================
# SELESAI
# ==========================================
print("\n" + "=" * 60)
print("MODELLING SELESAI!")
print("=" * 60)
print("\n📊 Untuk melihat hasil di MLflow Dashboard:")
print("   1. Buka terminal/command prompt")
print("   2. Jalankan: mlflow ui --port 5000")
print("   3. Buka browser: http://localhost:5000")
print("\n✓ Semua run tersimpan di folder: ./mlruns")
print(f"✓ Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")