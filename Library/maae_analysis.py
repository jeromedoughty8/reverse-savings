
"""
MAAE (Mean Asymmetric Absolute Error) Analysis
Compares standard MAE with business-weighted MAAE across models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import os


def calculate_maae(y_true, y_pred):
    """
    Calculate Mean Asymmetric Absolute Error.
    
    Penalizes over-prediction (loss) more than under-prediction (opportunity loss).
    - Over-prediction (pred > actual): Full penalty (1x) - Default risk
    - Under-prediction (pred < actual): Half penalty (0.5x) - Opportunity cost
    """
    errors = y_true - y_pred
    asymmetric_errors = np.where(
        y_pred > y_true,  # Over-prediction
        errors,           # Full penalty (1x)
        errors * 0.5      # Half penalty (0.5x)
    )
    return np.mean(np.abs(asymmetric_errors))


def create_mae_vs_maae_comparison(y_test, lr_pred, ridge_pred, xgb_pred, output_dir='data/regression_comparison'):
    """
    Create comprehensive MAE vs MAAE comparison for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("MAE VS MAAE COMPARISON ANALYSIS")
    print("=" * 70)
    print()
    
    # Calculate MAE and MAAE for each model
    models = {
        'Linear Regression': lr_pred,
        'Ridge Regression': ridge_pred,
        'XGBoost': xgb_pred
    }
    
    results = []
    
    for model_name, predictions in models.items():
        mae = mean_absolute_error(y_test, predictions)
        maae = calculate_maae(y_test, predictions)
        
        # Calculate over/under prediction counts
        over_pred = np.sum(predictions > y_test)
        under_pred = np.sum(predictions < y_test)
        total = len(y_test)
        
        # Calculate weighted impact
        over_errors = np.abs(y_test[predictions > y_test] - predictions[predictions > y_test])
        under_errors = np.abs(y_test[predictions <= y_test] - predictions[predictions <= y_test])
        
        avg_over_error = np.mean(over_errors) if len(over_errors) > 0 else 0
        avg_under_error = np.mean(under_errors) if len(under_errors) > 0 else 0
        
        results.append({
            'Model': model_name,
            'MAE': mae,
            'MAAE': maae,
            'Difference': mae - maae,
            'MAAE_Improvement_%': ((mae - maae) / mae * 100) if mae > 0 else 0,
            'Over_Predictions': over_pred,
            'Under_Predictions': under_pred,
            'Over_Pred_%': (over_pred / total * 100),
            'Under_Pred_%': (under_pred / total * 100),
            'Avg_Over_Error': avg_over_error,
            'Avg_Under_Error': avg_under_error
        })
    
    df = pd.DataFrame(results)
    
    # Print comparison table
    print("MAE VS MAAE COMPARISON TABLE:")
    print("-" * 70)
    print(df.to_string(index=False))
    print()
    
    print("INTERPRETATION:")
    print("-" * 70)
    print("• MAE: Standard metric - treats over/under prediction equally")
    print("• MAAE: Business-weighted metric - penalizes over-prediction 2x more")
    print("• Over-prediction = Financial Loss (default risk) - Full penalty")
    print("• Under-prediction = Opportunity Cost (lost revenue) - Half penalty")
    print()
    
    # Save CSV
    df.to_csv(f'{output_dir}/mae_vs_maae_comparison.csv', index=False)
    print(f"✓ Saved to: {output_dir}/mae_vs_maae_comparison.csv")
    print()
    
    # Create visualization
    create_mae_maae_visualizations(df, output_dir)
    
    return df


def create_mae_maae_visualizations(df, output_dir):
    """Create comprehensive visualizations for MAE vs MAAE."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = df['Model'].values
    mae_vals = df['MAE'].values
    maae_vals = df['MAAE'].values
    
    colors = ['steelblue', 'seagreen', 'darkorange']
    
    # 1. MAE vs MAAE Side-by-Side Comparison
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, mae_vals, width, label='MAE (Standard)', 
                           alpha=0.8, color=colors, edgecolor='black')
    bars2 = axes[0, 0].bar(x + width/2, maae_vals, width, label='MAAE (Business-Weighted)', 
                           alpha=0.8, color=['darkred', 'darkgreen', 'darkorange'], edgecolor='black')
    
    axes[0, 0].set_ylabel('Error ($)', fontweight='bold', fontsize=12)
    axes[0, 0].set_title('MAE vs MAAE Comparison\n(Lower is Better)', fontweight='bold', fontsize=14)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, fontsize=10)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. MAAE Improvement Percentage
    improvement = df['MAAE_Improvement_%'].values
    bars3 = axes[0, 1].bar(x, improvement, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_ylabel('MAAE Improvement (%)', fontweight='bold', fontsize=12)
    axes[0, 1].set_title('Business-Weighted Error Reduction\n(Higher is Better)', 
                         fontweight='bold', fontsize=14)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, fontsize=10)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Over vs Under Prediction Distribution
    over_preds = df['Over_Pred_%'].values
    under_preds = df['Under_Pred_%'].values
    
    bars4 = axes[1, 0].bar(x - width/2, over_preds, width, label='Over-Predictions (Risk)', 
                           alpha=0.8, color='darkred', edgecolor='black')
    bars5 = axes[1, 0].bar(x + width/2, under_preds, width, label='Under-Predictions (Opportunity)', 
                           alpha=0.8, color='darkgreen', edgecolor='black')
    
    axes[1, 0].set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    axes[1, 0].set_title('Prediction Direction Distribution', fontweight='bold', fontsize=14)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, fontsize=10)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Average Error by Prediction Type
    avg_over = df['Avg_Over_Error'].values
    avg_under = df['Avg_Under_Error'].values
    
    bars6 = axes[1, 1].bar(x - width/2, avg_over, width, label='Avg Over-Prediction Error', 
                           alpha=0.8, color='darkred', edgecolor='black')
    bars7 = axes[1, 1].bar(x + width/2, avg_under, width, label='Avg Under-Prediction Error', 
                           alpha=0.8, color='darkgreen', edgecolor='black')
    
    axes[1, 1].set_ylabel('Average Error ($)', fontweight='bold', fontsize=12)
    axes[1, 1].set_title('Average Error by Prediction Direction', fontweight='bold', fontsize=14)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, fontsize=10)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bars in [bars6, bars7]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle('MAE vs MAAE: Business-Weighted Error Analysis', 
                 fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mae_vs_maae_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to: {output_dir}/mae_vs_maae_comparison.png")
    print()


if __name__ == "__main__":
    print("MAAE Analysis Module - Import into main pipeline")
