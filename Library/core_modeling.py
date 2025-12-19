"""
Reverse Savings Credit System - Week 3
Core Modeling & Baseline Comparison
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Library.model_evaluation import ConfusionMatrixAnalyzer

def load_processed_data(file_path='data/models_output/processed_data.npz'):
    """Load the preprocessed data from Week 2."""
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("CORE MODELING & BASELINE COMPARISON")
    print("=" * 70)
    print()

    print("Step 1: Loading Preprocessed Data")
    print("-" * 70)

    data = np.load(file_path, allow_pickle=True)

    X_train = data['X_train']
    X_test = data['X_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_class_train = data['y_class_train']
    y_class_test = data['y_class_test']
    y_reg_train = data['y_reg_train']
    y_reg_test = data['y_reg_test']
    feature_names = data['feature_names']

    # Create validation set from training set (20% of training data)
    X_train_final, X_val, X_train_scaled_final, X_val_scaled, y_class_train_final, y_class_val = train_test_split(
        X_train, X_train_scaled, y_class_train, test_size=0.2, random_state=42, stratify=y_class_train
    )

    print(f"✓ Loaded training set: {len(X_train_final)} samples")
    print(f"✓ Loaded validation set: {len(X_val)} samples")
    print(f"✓ Loaded test set: {len(X_test)} samples")
    print(f"✓ Features: {len(feature_names)} columns")
    print()

    return (X_train_final, X_val, X_test, X_train_scaled_final, X_val_scaled, X_test_scaled, 
            y_class_train_final, y_class_val, y_class_test, y_reg_train, y_reg_test, feature_names)

def train_baseline_model(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Train Logistic Regression baseline model."""
    print("Step 2: Training Baseline Model (Logistic Regression)")
    print("-" * 70)

    # Train model
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train, y_train)

    print("✓ Logistic Regression trained")
    print()

    # Predictions for all sets
    y_pred_train = lr_model.predict(X_train)
    y_pred_proba_train = lr_model.predict_proba(X_train)[:, 1]

    y_pred_val = lr_model.predict(X_val)
    y_pred_proba_val = lr_model.predict_proba(X_val)[:, 1]

    y_pred_test = lr_model.predict(X_test)
    y_pred_proba_test = lr_model.predict_proba(X_test)[:, 1]

    # Performance metrics for all sets
    print("BASELINE MODEL PERFORMANCE (TRAIN SET):")
    print("-" * 70)
    analyzer_train = ConfusionMatrixAnalyzer(y_train, y_pred_train, y_pred_proba_train)
    metrics_train = analyzer_train.print_business_interpretation()

    print("\nBASELINE MODEL PERFORMANCE (VALIDATION SET):")
    print("-" * 70)
    analyzer_val = ConfusionMatrixAnalyzer(y_val, y_pred_val, y_pred_proba_val)
    metrics_val = analyzer_val.print_business_interpretation()

    print("\nBASELINE MODEL PERFORMANCE (TEST SET):")
    print("-" * 70)
    analyzer_test = ConfusionMatrixAnalyzer(y_test, y_pred_test, y_pred_proba_test)
    metrics_test = analyzer_test.print_business_interpretation()

    # Overfitting check
    print("\nOVERFITTING ANALYSIS:")
    print("-" * 70)
    print(f"Accuracy  - Train: {metrics_train['accuracy']:.4f} | Val: {metrics_val['accuracy']:.4f} | Test: {metrics_test['accuracy']:.4f}")
    print(f"F1-Score  - Train: {metrics_train['f1_score']:.4f} | Val: {metrics_val['f1_score']:.4f} | Test: {metrics_test['f1_score']:.4f}")
    print(f"AUC-ROC   - Train: {metrics_train['auc_roc']:.4f} | Val: {metrics_val['auc_roc']:.4f} | Test: {metrics_test['auc_roc']:.4f}")

    train_val_gap = metrics_train['f1_score'] - metrics_val['f1_score']
    if train_val_gap > 0.05:
        print(f"⚠ Overfitting detected (Train-Val F1 gap: {train_val_gap:.4f})")
    else:
        print(f"✓ No significant overfitting (Train-Val F1 gap: {train_val_gap:.4f})")
    print()

    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    print("Top 10 Most Important Features (Logistic Regression):")
    print("-" * 70)
    print(feature_importance.head(10).to_string(index=False))
    print()

    return lr_model, y_pred_test, y_pred_proba_test, metrics_train, metrics_val, metrics_test

def train_xgboost_model(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Train XGBoost primary model."""
    print("Step 3: Training Primary Model (XGBoost)")
    print("-" * 70)

    # Calculate scale_pos_weight for imbalanced dataset
    # IMPROVED: Increase weight to penalize missing defaults more
    base_scale = (y_train == 0).sum() / (y_train == 1).sum()
    scale_pos_weight = base_scale * 1.5  # Amplify importance of catching defaults

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist'
    }

    print(f"Model parameters:")
    print(f"  - Max depth: {params['max_depth']}")
    print(f"  - Learning rate: {params['learning_rate']}")
    print(f"  - N estimators: {params['n_estimators']}")
    print(f"  - Scale pos weight: {scale_pos_weight:.2f} (handles class imbalance)")
    print()

    # Train model with validation monitoring
    # The base_score is explicitly set to 0.5 to avoid SHAP compatibility issues.
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        base_score=0.5,
        scale_pos_weight=scale_pos_weight,  # Apply the increased weight
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train, 
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  verbose=False)

    print("✓ XGBoost trained")
    print()

    # Predictions for all sets
    y_pred_proba_train = xgb_model.predict_proba(X_train)[:, 1]
    y_pred_proba_val = xgb_model.predict_proba(X_val)[:, 1]
    y_pred_proba_test = xgb_model.predict_proba(X_test)[:, 1]

    # IMPROVED: Optimize threshold for better recall
    # Default threshold is 0.5, but we can lower it to catch more defaults
    # This trades some precision for better recall (catching more bad loans)
    optimal_threshold = 0.35  # Lower threshold = higher recall

    y_pred_train = (y_pred_proba_train >= optimal_threshold).astype(int)
    y_pred_val = (y_pred_proba_val >= optimal_threshold).astype(int)
    y_pred_test = (y_pred_proba_test >= optimal_threshold).astype(int)

    print(f"Using optimized decision threshold: {optimal_threshold}")
    print(f"  → Lower threshold increases recall (catches more defaults)")
    print(f"  → Trade-off: May decrease precision (more false alarms)")
    print()

    # Performance metrics for all sets
    print("XGBOOST MODEL PERFORMANCE (TRAIN SET):")
    print("-" * 70)
    analyzer_train = ConfusionMatrixAnalyzer(y_train, y_pred_train, y_pred_proba_train)
    metrics_train = analyzer_train.print_business_interpretation()

    print("\nXGBOOST MODEL PERFORMANCE (VALIDATION SET):")
    print("-" * 70)
    analyzer_val = ConfusionMatrixAnalyzer(y_val, y_pred_val, y_pred_proba_val)
    metrics_val = analyzer_val.print_business_interpretation()

    print("\nXGBOOST MODEL PERFORMANCE (TEST SET):")
    print("-" * 70)
    analyzer_test = ConfusionMatrixAnalyzer(y_test, y_pred_test, y_pred_proba_test)
    metrics_test = analyzer_test.print_business_interpretation()

    # Overfitting check
    print("\nOVERFITTING ANALYSIS:")
    print("-" * 70)
    print(f"Accuracy  - Train: {metrics_train['accuracy']:.4f} | Val: {metrics_val['accuracy']:.4f} | Test: {metrics_test['accuracy']:.4f}")
    print(f"F1-Score  - Train: {metrics_train['f1_score']:.4f} | Val: {metrics_val['f1_score']:.4f} | Test: {metrics_test['f1_score']:.4f}")
    print(f"AUC-ROC   - Train: {metrics_train['auc_roc']:.4f} | Val: {metrics_val['auc_roc']:.4f} | Test: {metrics_test['auc_roc']:.4f}")

    train_val_gap = metrics_train['f1_score'] - metrics_val['f1_score']
    if train_val_gap > 0.05:
        print(f"⚠ Overfitting detected (Train-Val F1 gap: {train_val_gap:.4f})")
    else:
        print(f"✓ No significant overfitting (Train-Val F1 gap: {train_val_gap:.4f})")
    print()

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 Most Important Features (XGBoost):")
    print("-" * 70)
    print(feature_importance.head(10).to_string(index=False))
    print()

    return xgb_model, y_pred_test, y_pred_proba_test, metrics_train, metrics_val, metrics_test, feature_importance

def compare_models(lr_train_metrics, lr_val_metrics, lr_test_metrics, 
                   xgb_train_metrics, xgb_val_metrics, xgb_test_metrics, 
                   output_dir='data/comparisons'):
    """Compare baseline and XGBoost models across train/val/test sets."""
    print("Step 4: Model Comparison")
    print("-" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Comparison table for TEST SET
    comparison_test = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'False Positive Rate'],
        'Logistic Regression': [
            lr_test_metrics.get('accuracy', 0),
            lr_test_metrics.get('precision', 0),
            lr_test_metrics.get('recall', 0),
            lr_test_metrics.get('f1_score', 0),
            lr_test_metrics.get('auc_roc', 0),
            lr_test_metrics.get('false_positive_rate', 0)
        ],
        'XGBoost': [
            xgb_test_metrics.get('accuracy', 0),
            xgb_test_metrics.get('precision', 0),
            xgb_test_metrics.get('recall', 0),
            xgb_test_metrics.get('f1_score', 0),
            xgb_test_metrics.get('auc_roc', 0),
            xgb_test_metrics.get('false_positive_rate', 0)
        ]
    })

    comparison_test['Improvement (%)'] = ((comparison_test['XGBoost'] - comparison_test['Logistic Regression']) 
                                           / comparison_test['Logistic Regression'] * 100)

    # Full comparison table (Train/Val/Test)
    comparison_full = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'LR_Train': [lr_train_metrics.get(m, 0) for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']],
        'LR_Val': [lr_val_metrics.get(m, 0) for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']],
        'LR_Test': [lr_test_metrics.get(m, 0) for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']],
        'XGB_Train': [xgb_train_metrics.get(m, 0) for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']],
        'XGB_Val': [xgb_val_metrics.get(m, 0) for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']],
        'XGB_Test': [xgb_test_metrics.get(m, 0) for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
    })

    print("\nMODEL COMPARISON (TEST SET):")
    print(comparison_test.to_string(index=False))
    print()

    print("\nFULL COMPARISON (TRAIN/VAL/TEST):")
    print(comparison_full.to_string(index=False))
    print()

    # Determine winner
    if xgb_test_metrics['f1_score'] > lr_test_metrics['f1_score']:
        print("🏆 WINNER: XGBoost")
        print(f"   F1-Score improvement: {(xgb_test_metrics['f1_score'] - lr_test_metrics['f1_score'])*100:.2f} percentage points")
    else:
        print("🏆 WINNER: Logistic Regression")

    print()

    # Save comparisons
    comparison_test.to_csv(f'{output_dir}/model_comparison_test.csv', index=False)
    comparison_full.to_csv(f'{output_dir}/model_comparison_full.csv', index=False)
    print(f"✓ Test comparison saved to: {output_dir}/model_comparison_test.csv")
    print(f"✓ Full comparison saved to: {output_dir}/model_comparison_full.csv")
    print()

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_to_plot = comparison_test[comparison_test['Metric'] != 'False Positive Rate'].copy()
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    bars1 = ax.bar(x - width/2, metrics_to_plot['Logistic Regression'], width, 
                   label='Logistic Regression', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, metrics_to_plot['XGBoost'], width, 
                   label='XGBoost', alpha=0.8, color='darkorange')

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison (Test Set)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot['Metric'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_dir}/model_comparison.png")
    plt.close()

    return comparison_test, comparison_full

def plot_feature_importance(feature_importance, output_dir='data/XGBoost'):
    """Plot XGBoost feature importance."""
    plt.figure(figsize=(10, 8))

    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontweight='bold')
    plt.title('Top 15 Features (XGBoost)', fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to: {output_dir}/feature_importance.png")
    plt.close()

def week3_main():
    """Execute Week 3 workflow."""
    # Load data with validation set
    (X_train, X_val, X_test, X_train_scaled, X_val_scaled, X_test_scaled, 
     y_class_train, y_class_val, y_class_test, y_reg_train, y_reg_test, feature_names) = load_processed_data()

    # Train baseline (Logistic Regression uses scaled features)
    lr_model, lr_pred, lr_proba, lr_train_metrics, lr_val_metrics, lr_test_metrics = train_baseline_model(
        X_train_scaled, X_val_scaled, X_test_scaled, y_class_train, y_class_val, y_class_test, feature_names
    )

    # Train XGBoost (XGBoost uses unscaled features - it's tree-based)
    xgb_model, xgb_pred, xgb_proba, xgb_train_metrics, xgb_val_metrics, xgb_test_metrics, feature_importance = train_xgboost_model(
        X_train, X_val, X_test, y_class_train, y_class_val, y_class_test, feature_names
    )

    # Compare models
    comparison_test, comparison_full = compare_models(
        lr_train_metrics, lr_val_metrics, lr_test_metrics,
        xgb_train_metrics, xgb_val_metrics, xgb_test_metrics
    )

    # Plot feature importance
    plot_feature_importance(feature_importance)

    # MAAE Classification Analysis - Critical Business Metric
    print("\n" + "=" * 70)
    print("GENERATING MAAE CLASSIFICATION COST ANALYSIS")
    print("=" * 70)
    from Library.maae_classification_analysis import create_maae_classification_comparison
    maae_class_comparison_df = create_maae_classification_comparison(
        y_class_test, lr_pred, xgb_pred, fn_cost=2.0, fp_cost=1.0
    )

    print("\n💡 KEY INSIGHT:")
    print("   MAAE penalizes False Negatives (missed defaults) 2x more than False Positives")
    print("   Lower Total Asymmetric Cost = Better business outcome (less financial loss)")
    print()

    # Save models
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_model, 'models/logistic_regression_baseline.pkl')
    joblib.dump(xgb_model, 'models/xgboost_primary.pkl')
    print(f"\n✓ Models saved to: models/")
    print()

    print("=" * 70)
    print("CORE MODELING COMPLETE! ✓")
    print("=" * 70)
    print("Deliverables:")
    print("  ✓ Logistic Regression baseline trained")
    print("  ✓ XGBoost primary model trained")
    print("  ✓ Train/Val/Test metrics calculated")
    print("  ✓ Overfitting analysis completed")
    print("  ✓ Confusion matrices generated")
    print("  ✓ Performance metrics calculated")
    print("  ✓ Model comparison completed")
    print("  ✓ Feature importance analyzed")
    print()
    print("Next: Week 4 - Model Optimization & Tuning")
    print("=" * 70)

if __name__ == "__main__":
    week3_main()