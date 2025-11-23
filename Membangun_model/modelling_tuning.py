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
    matthews_corrcoef, cohen_kappa_score, 
    classification_report, balanced_accuracy_score,
    log_loss, make_scorer
)
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
    
    base_path = "../preprosessing/data/output"
    
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
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filename.endswith('.npy'):
            data[key] = np.load(filepath)
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data[key] = pickle.load(f)
    
    print(f"✅ Data loaded successfully:")
    print(f"   - X_train: {data['X_train'].shape}")
    print(f"   - X_test: {data['X_test'].shape}")
    print(f"   - Features: {len(data['feature_names'])}")
    
    return data


def calculate_comprehensive_metrics(model, X_train, X_test, y_train, y_test):
    """
    Calculate comprehensive metrics dengan tambahan custom metrics
    """
    print("\n📊 Calculating comprehensive metrics...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix components
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Comprehensive metrics
    metrics = {
        # Basic accuracy metrics
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        
        # Precision-Recall-F1
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0),
        
        # ROC & Probability
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'log_loss': log_loss(y_test, y_test_proba),
        
        # Statistical correlation
        'matthews_corrcoef': matthews_corrcoef(y_test, y_test_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_test_pred),
        
        # Confusion matrix derived
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        
        # Rates
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
        
        # Model quality indicators
        'overfitting_gap': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred),
    }
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    # Additional CV metrics
    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    metrics['cv_f1_mean'] = cv_f1.mean()
    metrics['cv_f1_std'] = cv_f1.std()
    
    print("✅ Comprehensive metrics calculated")
    
    return metrics, y_test_pred, y_test_proba


def create_advanced_visualizations(model, X_test, y_test, y_pred, y_proba, 
                                   feature_names, best_params, cv_results):
    """
    Create advanced visualizations
    """
    print("\n🎨 Creating advanced visualizations...")
    
    os.makedirs("artifacts/tuning", exist_ok=True)
    
    # 1. Enhanced Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
                xticklabels=['Bad Wine (0)', 'Good Wine (1)'],
                yticklabels=['Bad Wine (0)', 'Good Wine (1)'],
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix\n(Percentage per True Class)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add counts
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'n={cm[i,j]}', 
                    ha='center', va='center', fontsize=10, color='black')
    
    plt.tight_layout()
    plt.savefig('artifacts/tuning/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance with Error Bars
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('viridis', len(fi_df))
    bars = plt.barh(range(len(fi_df)), fi_df['importance'].values, color=colors)
    plt.yticks(range(len(fi_df)), fi_df['feature'].values)
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.title('Feature Importance Analysis\n(All Features)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('artifacts/tuning/feature_importance_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/tuning/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/tuning/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Hyperparameter Impact Analysis
    results_df = pd.DataFrame(cv_results)
    
    # n_estimators impact
    if 'param_n_estimators' in results_df.columns:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        n_est_data = results_df.groupby('param_n_estimators')['mean_test_score'].agg(['mean', 'std'])
        plt.errorbar(n_est_data.index, n_est_data['mean'], yerr=n_est_data['std'], 
                    marker='o', linewidth=2, markersize=8, capsize=5)
        plt.xlabel('n_estimators', fontsize=12)
        plt.ylabel('Mean CV Score', fontsize=12)
        plt.title('Impact of n_estimators', fontsize=13, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # max_depth impact
        plt.subplot(1, 2, 2)
        depth_data = results_df.groupby('param_max_depth')['mean_test_score'].agg(['mean', 'std'])
        plt.errorbar(range(len(depth_data)), depth_data['mean'], yerr=depth_data['std'],
                    marker='s', linewidth=2, markersize=8, capsize=5)
        plt.xticks(range(len(depth_data)), [str(x) for x in depth_data.index])
        plt.xlabel('max_depth', fontsize=12)
        plt.ylabel('Mean CV Score', fontsize=12)
        plt.title('Impact of max_depth', fontsize=13, fontweight='bold')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('artifacts/tuning/hyperparameter_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Learning Curve (simplified)
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_test, y_test, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                     alpha=0.1, color='red')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Learning Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/tuning/learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Detailed Classification Report
    report = classification_report(y_test, y_pred, 
                                   target_names=['Bad Wine (0)', 'Good Wine (1)'],
                                   digits=4)
    with open('artifacts/tuning/classification_report_detailed.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("BEST HYPERPARAMETERS\n")
        f.write("=" * 60 + "\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    print("✅ Advanced visualizations saved to artifacts/tuning/")
    
    return [
        'artifacts/tuning/confusion_matrix_normalized.png',
        'artifacts/tuning/feature_importance_detailed.png',
        'artifacts/tuning/roc_curve.png',
        'artifacts/tuning/precision_recall_curve.png',
        'artifacts/tuning/hyperparameter_impact.png',
        'artifacts/tuning/learning_curve.png',
        'artifacts/tuning/classification_report_detailed.txt'
    ]


def hyperparameter_tuning():
    """
    Main hyperparameter tuning function with comprehensive MLflow tracking
    """
    print("=" * 70)
    print("🚀 ADVANCED HYPERPARAMETER TUNING")
    print("=" * 70)
    
    # Setup MLflow
    setup_mlflow()
    get_or_create_experiment("wine_quality_classification")
    
    # Load data
    data = load_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    total_combinations = (
        len(param_grid['n_estimators']) *
        len(param_grid['max_depth']) *
        len(param_grid['min_samples_split']) *
        len(param_grid['min_samples_leaf']) *
        len(param_grid['max_features']) *
        len(param_grid['bootstrap'])
    )
    
    print(f"\n📋 Parameter Grid:")
    for param, values in param_grid.items():
        print(f"   - {param}: {values}")
    print(f"\n🔢 Total combinations to test: {total_combinations}")
    
    # Initialize base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Setup GridSearchCV
    print("\n⚙️  Setting up GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # Use F1 as primary metric
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="hyperparameter_tuning_gridsearch") as parent_run:
        print(f"\n🎯 Parent Run ID: {parent_run.info.run_id}")
        
        # Log tuning metadata
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring_metric", "f1")
        mlflow.log_param("total_combinations", total_combinations)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Fit GridSearch
        print("\n🔍 Starting GridSearchCV (this may take a while)...")
        grid_search.fit(X_train, y_train)
        print("\n✅ GridSearchCV completed!")
        
        # Get best results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        print("\n" + "=" * 70)
        print("🏆 BEST PARAMETERS FOUND")
        print("=" * 70)
        for param, value in best_params.items():
            print(f"   ✅ {param}: {value}")
            mlflow.log_param(f"best_{param}", value)
        
        print(f"\n📊 Best CV F1-Score: {best_score:.4f}")
        mlflow.log_metric("best_cv_f1_score", best_score)
        
        # Log top 10 configurations as nested runs
        print("\n📝 Logging top 10 configurations...")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df.sort_values('rank_test_score')
        
        for idx, row in results_df.head(10).iterrows():
            rank = int(row['rank_test_score'])
            with mlflow.start_run(run_name=f"config_rank_{rank}", nested=True):
                # Log parameters
                params = row['params']
                for param, value in params.items():
                    mlflow.log_param(param, value)
                
                # Log scores
                mlflow.log_metric("mean_test_score", float(row['mean_test_score']))
                mlflow.log_metric("std_test_score", float(row['std_test_score']))
                mlflow.log_metric("mean_train_score", float(row['mean_train_score']))
                mlflow.log_metric("rank", float(rank))
                
                print(f"   ✅ Logged config rank {rank}")
        
        # Evaluate best model
        print("\n" + "=" * 70)
        print("📊 EVALUATING BEST MODEL")
        print("=" * 70)
        
        metrics, y_pred, y_proba = calculate_comprehensive_metrics(
            best_model, X_train, X_test, y_train, y_test
        )
        
        # Log all metrics
        print("\n📈 Logging comprehensive metrics...")
        for name, value in metrics.items():
            mlflow.log_metric(f"final_{name}", float(value))
            print(f"   ✅ {name}: {value:.6f}")
        
        # Create and log visualizations
        artifact_paths = create_advanced_visualizations(
            best_model, X_test, y_test, y_pred, y_proba,
            feature_names, best_params, grid_search.cv_results_
        )
        
        print("\n📦 Logging artifacts...")
        for path in artifact_paths:
            mlflow.log_artifact(path)
            print(f"   ✅ {os.path.basename(path)}")
        
        # Log scaler
        mlflow.log_artifact('../preprosessing/data/output/scaler.pkl')
        
        # Save and log best model
        print("\n💾 Saving best model...")
        
        # Save locally
        model_path = 'best_model_tuned.pkl'
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        
        # Log to MLflow Model Registry
        try:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                registered_model_name="wine_quality_rf_tuned"
            )
            print("   ✅ Model registered in MLflow Model Registry")
        except Exception as e:
            print(f"   ⚠️  Model registration skipped: {e}")
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model"
            )
        
        # Create comprehensive summary report
        print("\n📄 Creating summary report...")
        
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("HYPERPARAMETER TUNING SUMMARY REPORT")
        summary_lines.append("=" * 70)
        summary_lines.append(f"\nDate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"MLflow Run ID: {parent_run.info.run_id}")
        summary_lines.append(f"\nTuning Configuration:")
        summary_lines.append(f"  - Method: GridSearchCV")
        summary_lines.append(f"  - Cross-Validation Folds: 5")
        summary_lines.append(f"  - Scoring Metric: F1-Score")
        summary_lines.append(f"  - Total Configurations Tested: {total_combinations}")
        
        summary_lines.append(f"\n" + "=" * 70)
        summary_lines.append("BEST HYPERPARAMETERS")
        summary_lines.append("=" * 70)
        for param, value in best_params.items():
            summary_lines.append(f"  {param:25s}: {value}")
        
        summary_lines.append(f"\n" + "=" * 70)
        summary_lines.append("PERFORMANCE METRICS (Test Set)")
        summary_lines.append("=" * 70)
        
        key_metrics = [
            'test_accuracy', 'balanced_accuracy', 'precision', 'recall',
            'f1_score', 'roc_auc', 'matthews_corrcoef', 'cohen_kappa',
            'specificity', 'sensitivity'
        ]
        
        for metric in key_metrics:
            if metric in metrics:
                summary_lines.append(f"  {metric:25s}: {metrics[metric]:.6f}")
        
        summary_lines.append(f"\n" + "=" * 70)
        summary_lines.append("CROSS-VALIDATION RESULTS")
        summary_lines.append("=" * 70)
        summary_lines.append(f"  CV Accuracy Mean: {metrics['cv_accuracy_mean']:.6f} ± {metrics['cv_accuracy_std']:.6f}")
        summary_lines.append(f"  CV F1 Mean:       {metrics['cv_f1_mean']:.6f} ± {metrics['cv_f1_std']:.6f}")
        
        summary_lines.append(f"\n" + "=" * 70)
        summary_lines.append("MODEL QUALITY INDICATORS")
        summary_lines.append("=" * 70)
        summary_lines.append(f"  Overfitting Gap:  {metrics['overfitting_gap']:.6f}")
        summary_lines.append(f"  Log Loss:         {metrics['log_loss']:.6f}")
        
        summary_lines.append(f"\n" + "=" * 70)
        summary_lines.append("CONFUSION MATRIX DETAILS")
        summary_lines.append("=" * 70)
        summary_lines.append(f"  True Positives:   {metrics['true_positives']}")
        summary_lines.append(f"  True Negatives:   {metrics['true_negatives']}")
        summary_lines.append(f"  False Positives:  {metrics['false_positives']}")
        summary_lines.append(f"  False Negatives:  {metrics['false_negatives']}")
        summary_lines.append(f"  False Positive Rate: {metrics['false_positive_rate']:.6f}")
        summary_lines.append(f"  False Negative Rate: {metrics['false_negative_rate']:.6f}")
        
        summary_lines.append(f"\n" + "=" * 70)
        summary_lines.append("FEATURE IMPORTANCE (Top 5)")
        summary_lines.append("=" * 70)
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in fi_df.head(5).iterrows():
            summary_lines.append(f"  {row['feature']:25s}: {row['importance']:.6f}")
        
        summary_lines.append(f"\n" + "=" * 70)
        summary_lines.append("END OF REPORT")
        summary_lines.append("=" * 70)
        
        # Save summary
        summary_path = 'artifacts/tuning/tuning_summary_report.txt'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        mlflow.log_artifact(summary_path)
        print(f"   ✅ Summary report saved")
        
        # Set tags
        mlflow.set_tag("tuning_completed", "true")
        mlflow.set_tag("best_model_logged", "true")
        mlflow.set_tag("developer", "Saepulloh")
        mlflow.set_tag("algorithm", "RandomForest_Tuned")
        mlflow.set_tag("dataset", "Wine_Quality")
        mlflow.set_tag("model_version", "v2.0_tuned")
        
        # Print final summary
        print("\n" + "=" * 70)
        print("🎉 HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📊 Key Results:")
        print(f"   - Best CV F1-Score:    {best_score:.4f}")
        print(f"   - Test Accuracy:       {metrics['test_accuracy']:.4f}")
        print(f"   - Test F1-Score:       {metrics['f1_score']:.4f}")
        print(f"   - ROC-AUC:            {metrics['roc_auc']:.4f}")
        print(f"   - Matthews Corr:      {metrics['matthews_corrcoef']:.4f}")
        
        print(f"\n🌐 View detailed results:")
        print(f"   MLflow UI: {mlflow.get_tracking_uri()}")
        print(f"   Run ID: {parent_run.info.run_id}")
        
        print(f"\n💾 Best model saved as: {model_path}")
        print(f"📄 Full report: {summary_path}")
        
        return best_model, metrics


if __name__ == "__main__":
    try:
        best_model, metrics = hyperparameter_tuning()
        print("\n✅ All tasks completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)