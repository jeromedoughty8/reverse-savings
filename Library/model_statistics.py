"""
Model Statistics and Loss Function Analysis
Generates detailed performance tables for Logistic Regression and XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
import os

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, model_name):
    """
    Calculate all relevant metrics including loss functions.

    Standard Convention:
    - y_true, y_pred: 0 = Repays (negative), 1 = Default (positive)
    """

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    # Calculate confusion matrix with new structure
    # Row 1 (predicted=1): TP, FP | Row 2 (predicted=0): FN, TN
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fp, fn, tn = cm.ravel()

    # Loss functions
    logloss = log_loss(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)

    # Additional metrics
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Business metrics
    total_approved = tp + fn
    total_rejected = tn + fp
    default_rate = fn / total_approved if total_approved > 0 else 0

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'Specificity': specificity,
        'Log Loss': logloss,
        'Brier Score': brier_score,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
        'True Positives': int(tp),
        'True Negatives': int(tn),
        'False Positives': int(fp),
        'False Negatives': int(fn),
        'Total Approved': int(total_approved),
        'Default Rate (%)': default_rate * 100
    }

def create_loss_function_table(lr_metrics, xgb_metrics, output_dir):
    """Create detailed loss function comparison table."""

    loss_data = {
        'Loss Function': ['Log Loss', 'Brier Score', 'Mean Squared Error (Probabilities)'],
        'Logistic Regression': [
            lr_metrics['Log Loss'],
            lr_metrics['Brier Score'],
            mean_squared_error([0, 1], [lr_metrics['Log Loss'], lr_metrics['Brier Score']])
        ],
        'XGBoost': [
            xgb_metrics['Log Loss'],
            xgb_metrics['Brier Score'],
            mean_squared_error([0, 1], [xgb_metrics['Log Loss'], xgb_metrics['Brier Score']])
        ]
    }

    df = pd.DataFrame(loss_data)
    df['Improvement (%)'] = ((df['Logistic Regression'] - df['XGBoost']) / df['Logistic Regression'] * 100)

    # Save CSV
    df.to_csv(f'{output_dir}/loss_functions_comparison.csv', index=False)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    # Format values for display
    display_data = df.copy()
    for col in ['Logistic Regression', 'XGBoost']:
        display_data[col] = display_data[col].apply(lambda x: f'{x:.6f}')
    display_data['Improvement (%)'] = display_data['Improvement (%)'].apply(lambda x: f'{x:.2f}%')

    table = ax.table(cellText=display_data.values, colLabels=display_data.columns,
                     cellLoc='center', loc='center',
                     colColours=['#4CAF50']*len(display_data.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(display_data.columns)):
        table[(0, i)].set_facecolor('#2E7D32')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Loss Function Comparison - Lower is Better',
              fontweight='bold', fontsize=14, pad=20)
    plt.savefig(f'{output_dir}/loss_functions_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    return df

def create_comprehensive_metrics_table(lr_metrics, xgb_metrics, output_dir):
    """Create comprehensive metrics table."""

    metrics_order = [
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC',
        'Specificity', 'Log Loss', 'Brier Score',
        'False Positive Rate', 'False Negative Rate'
    ]

    comparison_data = []
    for metric in metrics_order:
        comparison_data.append({
            'Metric': metric,
            'Logistic Regression': lr_metrics[metric],
            'XGBoost': xgb_metrics[metric]
        })

    df = pd.DataFrame(comparison_data)

    # Calculate improvement (for loss functions, negative is better)
    loss_metrics = ['Log Loss', 'Brier Score', 'False Positive Rate', 'False Negative Rate']

    def calc_improvement(row):
        metric = row['Metric']
        lr_val = row['Logistic Regression']
        xgb_val = row['XGBoost']

        if metric in loss_metrics:
            # For loss functions, lower is better
            return ((lr_val - xgb_val) / lr_val * 100)
        else:
            # For performance metrics, higher is better
            return ((xgb_val - lr_val) / lr_val * 100)

    df['Improvement (%)'] = df.apply(calc_improvement, axis=1)

    # Save CSV
    df.to_csv(f'{output_dir}/comprehensive_metrics.csv', index=False)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Format values for display
    display_data = df.copy()
    for col in ['Logistic Regression', 'XGBoost']:
        display_data[col] = display_data[col].apply(lambda x: f'{x:.6f}' if x < 1 else f'{x:.2f}')
    display_data['Improvement (%)'] = display_data['Improvement (%)'].apply(lambda x: f'{x:+.2f}%')

    table = ax.table(cellText=display_data.values, colLabels=display_data.columns,
                     cellLoc='center', loc='center',
                     colColours=['#1976D2']*len(display_data.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(display_data.columns)):
        table[(0, i)].set_facecolor('#0D47A1')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight improvements (green for positive, red for negative)
    for i in range(1, len(display_data) + 1):
        imp_value = df.iloc[i-1]['Improvement (%)']
        if imp_value > 0:
            table[(i, 3)].set_facecolor('#C8E6C9')
        else:
            table[(i, 3)].set_facecolor('#FFCDD2')

    plt.title('Comprehensive Model Performance Metrics',
              fontweight='bold', fontsize=16, pad=20)
    plt.savefig(f'{output_dir}/comprehensive_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    return df

def create_confusion_matrix_table(lr_metrics, xgb_metrics, output_dir):
    """Create confusion matrix comparison table."""

    cm_data = {
        'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives',
                   'Total Approved', 'Default Rate (%)'],
        'Logistic Regression': [
            lr_metrics['True Positives'],
            lr_metrics['True Negatives'],
            lr_metrics['False Positives'],
            lr_metrics['False Negatives'],
            lr_metrics['Total Approved'],
            lr_metrics['Default Rate (%)']
        ],
        'XGBoost': [
            xgb_metrics['True Positives'],
            xgb_metrics['True Negatives'],
            xgb_metrics['False Positives'],
            xgb_metrics['False Negatives'],
            xgb_metrics['Total Approved'],
            xgb_metrics['Default Rate (%)']
        ]
    }

    df = pd.DataFrame(cm_data)

    # Save CSV
    df.to_csv(f'{output_dir}/confusion_matrix_comparison.csv', index=False)

    return df

def create_individual_model_visualizations(metrics, model_name, output_dir):
    """Create visualizations for individual model."""

    # 1. Performance metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    perf_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
    values = [metrics[m] for m in perf_metrics]

    bars = ax.bar(perf_metrics, values, color='steelblue', alpha=0.8)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'{model_name} - Performance Metrics', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_data = np.array([
        [metrics['True Positives'], metrics['False Positives']],
        [metrics['False Negatives'], metrics['True Negatives']]
    ])

    # Create labels (TP/FP first row, FN/TN second row)
    labels = np.array([
        [f'TP\n{metrics["True Positives"]}', f'FP\n{metrics["False Positives"]}'],
        [f'FN\n{metrics["False Negatives"]}', f'TN\n{metrics["True Negatives"]}']
    ])

    sns.heatmap(cm_data, annot=labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=['Defaults (1)', 'Repays (0)'],
                yticklabels=['Predicted: Default (1)', 'Predicted: No Default (0)'])
    ax.set_title(f'{model_name} - Confusion Matrix', fontweight='bold', fontsize=14)
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_xlabel('Predicted', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def load_synthetic_data(file_path=None):
    """Load the synthetic dataset - auto-detect the most recent or largest dataset."""
    print("Loading synthetic dataset...")
    print("-" * 70)

    # If no path specified, find the most appropriate dataset
    if file_path is None:
        possible_files = [
            'data/synthetic/synthetic_credit_data_5000K.csv',  # 5M samples (default)
            'data/synthetic/synthetic_credit_data_1000K.csv',  # 1M samples
            'data/synthetic/synthetic_credit_data_500K.csv',   # 500K samples
            'data/synthetic/synthetic_credit_data.csv'         # Fallback
        ]

        for candidate_path in possible_files:
            if os.path.exists(candidate_path):
                file_path = candidate_path
                break

    if file_path is None or not os.path.exists(file_path):
        print(f"❌ Error: Dataset not found")
        print("Please run data generation first (python main.py all)")
        return None

    df = pd.read_csv(file_path)
    n_samples = len(df)
    print(f"✓ Loaded dataset: {n_samples:,} samples from {file_path}")
    
    # Update display message based on actual size
    if n_samples >= 1000000:
        print(f"  └─ Large-scale dataset: {n_samples/1000000:.1f}M samples")
    elif n_samples >= 1000:
        print(f"  └─ Dataset: {n_samples/1000:.0f}K samples")
    print()

    return df


def generate_model_statistics(lr_output_dir='data/LogisticRegression',
                             xgb_output_dir='data/XGBoost',
                             comparison_dir='data/comparisons'):
    """Generate all statistics tables for both models."""

    print("=" * 70)
    print("GENERATING COMPREHENSIVE MODEL STATISTICS")
    print("=" * 70)
    print()

    # Load data and models
    data = np.load('data/models_output/processed_data.npz', allow_pickle=True)
    X_test_scaled = data['X_test_scaled']
    y_class_test = data['y_class_test']

    lr_model = joblib.load('models/logistic_regression_baseline.pkl')
    xgb_model = joblib.load('models/xgboost_primary.pkl')

    # Get predictions
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    xgb_pred = xgb_model.predict(data['X_test'])
    xgb_proba = xgb_model.predict_proba(data['X_test'])[:, 1]

    # Calculate comprehensive metrics
    print("Calculating Logistic Regression metrics...")
    lr_metrics = calculate_comprehensive_metrics(y_class_test, lr_pred, lr_proba,
                                                 'Logistic Regression')

    print("Calculating XGBoost metrics...")
    xgb_metrics = calculate_comprehensive_metrics(y_class_test, xgb_pred, xgb_proba,
                                                  'XGBoost')
    print()

    # Create output directories
    os.makedirs(lr_output_dir, exist_ok=True)
    os.makedirs(xgb_output_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Generate individual model visualizations
    print("Creating Logistic Regression visualizations...")
    create_individual_model_visualizations(lr_metrics, 'Logistic Regression', lr_output_dir)
    print(f"✓ Saved: {lr_output_dir}/performance_metrics.png")
    print(f"✓ Saved: {lr_output_dir}/confusion_matrix.png")
    print()

    print("Creating XGBoost visualizations...")
    create_individual_model_visualizations(xgb_metrics, 'XGBoost', xgb_output_dir)
    print(f"✓ Saved: {xgb_output_dir}/performance_metrics.png")
    print(f"✓ Saved: {xgb_output_dir}/confusion_matrix.png")
    print()

    # Generate loss function comparison
    print("Creating loss function comparison table...")
    loss_df = create_loss_function_table(lr_metrics, xgb_metrics, comparison_dir)
    print(f"✓ Saved: {comparison_dir}/loss_functions_comparison.csv")
    print(f"✓ Saved: {comparison_dir}/loss_functions_table.png")
    print()

    # Generate comprehensive metrics comparison
    print("Creating comprehensive metrics table...")
    metrics_df = create_comprehensive_metrics_table(lr_metrics, xgb_metrics, comparison_dir)
    print(f"✓ Saved: {comparison_dir}/comprehensive_metrics.csv")
    print(f"✓ Saved: {comparison_dir}/comprehensive_metrics_table.png")
    print()

    # Generate confusion matrix comparison
    print("Creating confusion matrix comparison...")
    cm_df = create_confusion_matrix_table(lr_metrics, xgb_metrics, comparison_dir)
    print(f"✓ Saved: {comparison_dir}/confusion_matrix_comparison.csv")
    print()

    # Save individual model statistics
    lr_stats = pd.DataFrame([lr_metrics])
    lr_stats.to_csv(f'{lr_output_dir}/model_statistics.csv', index=False)
    print(f"✓ Saved: {lr_output_dir}/model_statistics.csv")

    xgb_stats = pd.DataFrame([xgb_metrics])
    xgb_stats.to_csv(f'{xgb_output_dir}/model_statistics.csv', index=False)
    print(f"✓ Saved: {xgb_output_dir}/model_statistics.csv")
    print()

    print("=" * 70)
    print("STATISTICS GENERATION COMPLETE! ✓")
    print("=" * 70)
    print()
    print("Key Findings:")
    print(f"  • Log Loss - LR: {lr_metrics['Log Loss']:.6f}, XGB: {xgb_metrics['Log Loss']:.6f}")
    print(f"  • Brier Score - LR: {lr_metrics['Brier Score']:.6f}, XGB: {xgb_metrics['Brier Score']:.6f}")
    print(f"  • F1-Score - LR: {lr_metrics['F1-Score']:.4f}, XGB: {xgb_metrics['F1-Score']:.4f}")
    print(f"  • AUC-ROC - LR: {lr_metrics['AUC-ROC']:.4f}, XGB: {xgb_metrics['AUC-ROC']:.4f}")
    print()

if __name__ == "__main__":
    generate_model_statistics()