
"""
MAAE (Mean Asymmetric Absolute Error) Analysis for Classification
Analyzes the impact of asymmetric weighting on classification performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def calculate_asymmetric_classification_cost(y_true, y_pred, fn_cost=2.0, fp_cost=1.0):
    """
    Calculate asymmetric cost for classification.
    
    Args:
        y_true: Actual labels (0=Repays, 1=Default)
        y_pred: Predicted labels
        fn_cost: Cost multiplier for False Negatives (missed defaults)
        fp_cost: Cost multiplier for False Positives (rejected good customers)
    
    Returns:
        Total weighted cost
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    
    # Extract confusion matrix components
    tp = cm[0, 0]  # True Positives (correctly predicted defaults)
    fn = cm[0, 1]  # False Negatives (missed defaults) - FINANCIAL LOSS
    fp = cm[1, 0]  # False Positives (rejected good customers) - OPPORTUNITY COST
    tn = cm[1, 1]  # True Negatives (correctly predicted repays)
    
    # Calculate asymmetric cost
    # FN (missed default) = 2x penalty (financial loss)
    # FP (rejected good customer) = 1x penalty (opportunity cost)
    total_cost = (fn * fn_cost) + (fp * fp_cost)
    
    return {
        'total_cost': total_cost,
        'fn_cost': fn * fn_cost,
        'fp_cost': fp * fp_cost,
        'tp': tp,
        'tn': tn,
        'fn': fn,
        'fp': fp
    }


def create_maae_classification_comparison(y_test, lr_pred, xgb_pred, 
                                          fn_cost=2.0, fp_cost=1.0,
                                          output_dir='data/comparisons'):
    """
    Create comprehensive MAAE comparison for classification models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("MAAE CLASSIFICATION COST ANALYSIS")
    print("=" * 70)
    print()
    
    print(f"COST STRUCTURE:")
    print(f"  • False Negative (FN) - Missed Default: {fn_cost}x penalty (Financial Loss)")
    print(f"  • False Positive (FP) - Rejected Good Customer: {fp_cost}x penalty (Opportunity Cost)")
    print()
    
    # Calculate costs for each model
    models = {
        'Logistic Regression (MAAE-Weighted)': lr_pred,
        'XGBoost (MAAE-Weighted)': xgb_pred
    }
    
    results = []
    
    for model_name, predictions in models.items():
        cost_data = calculate_asymmetric_classification_cost(y_test, predictions, fn_cost, fp_cost)
        
        # Calculate standard metrics
        cm = confusion_matrix(y_test, predictions, labels=[1, 0])
        tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        total_samples = len(y_test)
        accuracy = (tp + tn) / total_samples
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate cost per error
        total_errors = fn + fp
        avg_cost_per_error = cost_data['total_cost'] / total_errors if total_errors > 0 else 0
        
        results.append({
            'Model': model_name,
            'Total_Asymmetric_Cost': cost_data['total_cost'],
            'FN_Cost_Component': cost_data['fn_cost'],
            'FP_Cost_Component': cost_data['fp_cost'],
            'True_Positives': tp,
            'True_Negatives': tn,
            'False_Negatives': fn,
            'False_Positives': fp,
            'FN_Rate_%': (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0,
            'FP_Rate_%': (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Avg_Cost_Per_Error': avg_cost_per_error
        })
    
    df = pd.DataFrame(results)
    
    # Print comparison table
    print("ASYMMETRIC COST COMPARISON TABLE:")
    print("-" * 70)
    print(df[['Model', 'Total_Asymmetric_Cost', 'FN_Cost_Component', 'FP_Cost_Component']].to_string(index=False))
    print()
    
    print("CONFUSION MATRIX BREAKDOWN:")
    print("-" * 70)
    print(df[['Model', 'True_Positives', 'True_Negatives', 'False_Negatives', 'False_Positives']].to_string(index=False))
    print()
    
    print("ERROR RATE ANALYSIS:")
    print("-" * 70)
    print(df[['Model', 'FN_Rate_%', 'FP_Rate_%', 'Accuracy', 'Recall']].to_string(index=False))
    print()
    
    print("INTERPRETATION:")
    print("-" * 70)
    print("• Total_Asymmetric_Cost: Business-weighted error (lower is better)")
    print("• FN_Cost_Component: Cost from missed defaults (financial loss)")
    print("• FP_Cost_Component: Cost from rejected good customers (opportunity loss)")
    print("• FN_Rate: % of actual defaults we missed (should be minimized)")
    print("• FP_Rate: % of good customers we rejected (opportunity cost)")
    print()
    
    # Determine winner
    best_model_idx = df['Total_Asymmetric_Cost'].idxmin()
    best_model = df.loc[best_model_idx]
    
    print("🏆 BEST MODEL (Lowest Asymmetric Cost):")
    print("-" * 70)
    print(f"  Model: {best_model['Model']}")
    print(f"  Total Asymmetric Cost: {best_model['Total_Asymmetric_Cost']:.0f}")
    print(f"  False Negatives: {best_model['False_Negatives']:.0f} (missed defaults)")
    print(f"  False Positives: {best_model['False_Positives']:.0f} (rejected good customers)")
    print()
    
    # Save CSV
    df.to_csv(f'{output_dir}/maae_classification_comparison.csv', index=False)
    print(f"✓ Saved to: {output_dir}/maae_classification_comparison.csv")
    print()
    
    # Create visualization
    create_maae_classification_visualizations(df, output_dir)
    
    return df


def create_maae_classification_visualizations(df, output_dir):
    """Create comprehensive visualizations for MAAE classification analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = df['Model'].values
    colors = ['steelblue', 'darkorange']
    
    # 1. Total Asymmetric Cost Comparison
    x = np.arange(len(models))
    
    bars1 = axes[0, 0].bar(x, df['Total_Asymmetric_Cost'], color=colors, alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Total Asymmetric Cost', fontweight='bold', fontsize=12)
    axes[0, 0].set_title('Total Business-Weighted Cost\n(Lower is Better)', fontweight='bold', fontsize=14)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. FN vs FP Cost Breakdown
    width = 0.35
    fn_costs = df['FN_Cost_Component'].values
    fp_costs = df['FP_Cost_Component'].values
    
    bars2 = axes[0, 1].bar(x - width/2, fn_costs, width, label='FN Cost (Financial Loss)', 
                           alpha=0.8, color='darkred', edgecolor='black')
    bars3 = axes[0, 1].bar(x + width/2, fp_costs, width, label='FP Cost (Opportunity Loss)', 
                           alpha=0.8, color='orange', edgecolor='black')
    
    axes[0, 1].set_ylabel('Cost Component', fontweight='bold', fontsize=12)
    axes[0, 1].set_title('Cost Breakdown by Error Type', fontweight='bold', fontsize=14)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, fontsize=10)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Error Rate Comparison
    fn_rates = df['FN_Rate_%'].values
    fp_rates = df['FP_Rate_%'].values
    
    bars4 = axes[1, 0].bar(x - width/2, fn_rates, width, label='FN Rate % (Missed Defaults)', 
                           alpha=0.8, color='darkred', edgecolor='black')
    bars5 = axes[1, 0].bar(x + width/2, fp_rates, width, label='FP Rate % (Rejected Good Customers)', 
                           alpha=0.8, color='orange', edgecolor='black')
    
    axes[1, 0].set_ylabel('Error Rate (%)', fontweight='bold', fontsize=12)
    axes[1, 0].set_title('Error Rates by Type\n(Lower is Better)', fontweight='bold', fontsize=14)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, fontsize=10)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Standard Performance Metrics
    accuracy = df['Accuracy'].values * 100
    precision = df['Precision'].values * 100
    recall = df['Recall'].values * 100
    
    x_metrics = np.arange(3)
    bars6 = axes[1, 1].bar(x_metrics - width/2, 
                           [accuracy[0], precision[0], recall[0]], 
                           width, label=models[0], alpha=0.8, color=colors[0], edgecolor='black')
    bars7 = axes[1, 1].bar(x_metrics + width/2, 
                           [accuracy[1], precision[1], recall[1]], 
                           width, label=models[1], alpha=0.8, color=colors[1], edgecolor='black')
    
    axes[1, 1].set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    axes[1, 1].set_title('Standard Classification Metrics\n(Higher is Better)', fontweight='bold', fontsize=14)
    axes[1, 1].set_xticks(x_metrics)
    axes[1, 1].set_xticklabels(['Accuracy', 'Precision', 'Recall'], fontsize=10)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    for bars in [bars6, bars7]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle('MAAE Classification Analysis: Business-Weighted Error Comparison', 
                 fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/maae_classification_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to: {output_dir}/maae_classification_comparison.png")
    print()


if __name__ == "__main__":
    print("MAAE Classification Analysis Module - Import into main pipeline")
