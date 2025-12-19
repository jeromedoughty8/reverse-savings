
"""
Regression Model Visualization and Analysis
Generates comprehensive graphs and tables for model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def create_comprehensive_visualizations(output_dir='data/regression_analysis'):
    """Generate all regression visualization graphs and tables."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed data
    data = np.load('data/processed_data.npz', allow_pickle=True)
    X_test = data['X_test']
    X_test_scaled = data['X_test_scaled']
    y_reg_test = data['y_reg_test']
    feature_names = data['feature_names']
    
    # Load models
    lr_model = joblib.load('models/regression/linear_regression.pkl')
    ridge_model = joblib.load('models/regression/ridge_regression.pkl')
    xgb_model = joblib.load('models/regression/xgboost_regression.pkl')
    
    # Get predictions
    lr_pred = lr_model.predict(X_test_scaled)
    ridge_pred = ridge_model.predict(X_test_scaled)
    xgb_pred = xgb_model.predict(X_test)
    
    # 1. Create comprehensive metrics table
    create_metrics_table(y_reg_test, lr_pred, ridge_pred, xgb_pred, output_dir)
    
    # 2. Predicted vs Actual scatter plots
    create_prediction_scatter_plots(y_reg_test, lr_pred, ridge_pred, xgb_pred, output_dir)
    
    # 3. Residual analysis plots
    create_residual_plots(y_reg_test, lr_pred, ridge_pred, xgb_pred, output_dir)
    
    # 4. Error distribution comparison
    create_error_distribution(y_reg_test, lr_pred, ridge_pred, xgb_pred, output_dir)
    
    # 5. Performance metrics bar chart
    create_metrics_comparison_chart(y_reg_test, lr_pred, ridge_pred, xgb_pred, output_dir)
    
    # 6. Feature importance comparison
    create_feature_importance_comparison(lr_model, xgb_model, feature_names, output_dir)
    
    # 7. Prediction error by loan amount range
    create_error_by_range_analysis(y_reg_test, lr_pred, ridge_pred, xgb_pred, output_dir)
    
    print(f"\n✓ All regression visualizations saved to: {output_dir}/")

def create_metrics_table(y_true, lr_pred, ridge_pred, xgb_pred, output_dir):
    """Create comprehensive metrics comparison table."""
    
    models = {
        'Linear Regression': lr_pred,
        'Ridge Regression': ridge_pred,
        'XGBoost Regressor': xgb_pred
    }
    
    metrics_data = []
    
    for model_name, predictions in models.items():
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
        residuals = y_true - predictions
        
        metrics_data.append({
            'Model': model_name,
            'MSE': f'${mse:,.2f}²',
            'RMSE': f'${rmse:,.2f}',
            'MAE': f'${mae:,.2f}',
            'R²': f'{r2:.4f}',
            'MAPE': f'{mape:.2f}%',
            'Mean Residual': f'${np.mean(residuals):.2f}',
            'Std Residual': f'${np.std(residuals):.2f}'
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Save as CSV
    df.to_csv(f'{output_dir}/regression_metrics_table.csv', index=False)
    
    # Create formatted table visualization
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colColours=['#4CAF50']*len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2E7D32')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F5E9')
    
    plt.title('Regression Models - Performance Metrics Comparison', 
              fontweight='bold', fontsize=14, pad=20)
    
    plt.savefig(f'{output_dir}/metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics table saved")

def create_prediction_scatter_plots(y_true, lr_pred, ridge_pred, xgb_pred, output_dir):
    """Create predicted vs actual scatter plots for all models."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = [
        ('Linear Regression', lr_pred, 'steelblue'),
        ('Ridge Regression', ridge_pred, 'seagreen'),
        ('XGBoost Regressor', xgb_pred, 'darkorange')
    ]
    
    for idx, (name, pred, color) in enumerate(models):
        r2 = r2_score(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        
        axes[idx].scatter(y_true, pred, alpha=0.5, s=20, color=color)
        axes[idx].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                       'r--', lw=2, label='Perfect Prediction', alpha=0.7)
        
        axes[idx].set_xlabel('Actual Loan Amount ($)', fontweight='bold', fontsize=11)
        axes[idx].set_ylabel('Predicted Loan Amount ($)', fontweight='bold', fontsize=11)
        axes[idx].set_title(f'{name}\nR² = {r2:.4f}, RMSE = ${rmse:.2f}', 
                           fontweight='bold', fontsize=12)
        axes[idx].legend(loc='upper left')
        axes[idx].grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(y_true, pred, 1)
        p = np.poly1d(z)
        axes[idx].plot(y_true, p(y_true), "g-", alpha=0.5, lw=2, label='Trend')
    
    plt.suptitle('Predicted vs Actual Loan Amount - Model Comparison', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Prediction scatter plots saved")

def create_residual_plots(y_true, lr_pred, ridge_pred, xgb_pred, output_dir):
    """Create residual distribution and Q-Q plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    models = [
        ('Linear Regression', lr_pred, 'steelblue'),
        ('Ridge Regression', ridge_pred, 'seagreen'),
        ('XGBoost Regressor', xgb_pred, 'darkorange')
    ]
    
    for idx, (name, pred, color) in enumerate(models):
        residuals = y_true - pred
        
        # Residual histogram
        axes[0, idx].hist(residuals, bins=40, alpha=0.7, color=color, edgecolor='black')
        axes[0, idx].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0, idx].set_xlabel('Residual ($)', fontweight='bold')
        axes[0, idx].set_ylabel('Frequency', fontweight='bold')
        axes[0, idx].set_title(f'{name}\nMean = ${np.mean(residuals):.2f}, Std = ${np.std(residuals):.2f}',
                              fontweight='bold')
        axes[0, idx].legend()
        axes[0, idx].grid(alpha=0.3)
        
        # Residual scatter plot
        axes[1, idx].scatter(pred, residuals, alpha=0.5, s=20, color=color)
        axes[1, idx].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, idx].set_xlabel('Predicted Amount ($)', fontweight='bold')
        axes[1, idx].set_ylabel('Residual ($)', fontweight='bold')
        axes[1, idx].set_title(f'{name}\nResiduals vs Predicted', fontweight='bold')
        axes[1, idx].grid(alpha=0.3)
    
    plt.suptitle('Residual Analysis - All Models', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Residual analysis saved")

def create_error_distribution(y_true, lr_pred, ridge_pred, xgb_pred, output_dir):
    """Create error distribution comparison."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models_data = [
        ('Linear Regression', np.abs(y_true - lr_pred), 'steelblue'),
        ('Ridge Regression', np.abs(y_true - ridge_pred), 'seagreen'),
        ('XGBoost Regressor', np.abs(y_true - xgb_pred), 'darkorange')
    ]
    
    for name, errors, color in models_data:
        ax.hist(errors, bins=50, alpha=0.5, label=name, color=color, edgecolor='black')
    
    ax.set_xlabel('Absolute Prediction Error ($)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax.set_title('Absolute Error Distribution - Model Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Error distribution saved")

def create_metrics_comparison_chart(y_true, lr_pred, ridge_pred, xgb_pred, output_dir):
    """Create bar chart comparing key metrics."""
    
    models = ['Linear\nRegression', 'Ridge\nRegression', 'XGBoost\nRegressor']
    
    rmse_vals = [
        np.sqrt(mean_squared_error(y_true, lr_pred)),
        np.sqrt(mean_squared_error(y_true, ridge_pred)),
        np.sqrt(mean_squared_error(y_true, xgb_pred))
    ]
    
    mae_vals = [
        mean_absolute_error(y_true, lr_pred),
        mean_absolute_error(y_true, ridge_pred),
        mean_absolute_error(y_true, xgb_pred)
    ]
    
    r2_vals = [
        r2_score(y_true, lr_pred),
        r2_score(y_true, ridge_pred),
        r2_score(y_true, xgb_pred)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    x = np.arange(len(models))
    colors = ['steelblue', 'seagreen', 'darkorange']
    
    # RMSE
    bars1 = axes[0].bar(x, rmse_vals, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('RMSE ($)', fontweight='bold', fontsize=11)
    axes[0].set_title('Root Mean Squared Error\n(Lower is Better)', fontweight='bold', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE
    bars2 = axes[1].bar(x, mae_vals, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('MAE ($)', fontweight='bold', fontsize=11)
    axes[1].set_title('Mean Absolute Error\n(Lower is Better)', fontweight='bold', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # R²
    bars3 = axes[2].bar(x, r2_vals, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('R² Score', fontweight='bold', fontsize=11)
    axes[2].set_title('R² Score\n(Higher is Better)', fontweight='bold', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models)
    axes[2].set_ylim([0, 1])
    axes[2].grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Regression Performance Metrics Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics comparison chart saved")

def create_feature_importance_comparison(lr_model, xgb_model, feature_names, output_dir):
    """Compare feature importance between Linear Regression and XGBoost."""
    
    # Linear Regression coefficients
    lr_importance = pd.DataFrame({
        'feature': feature_names,
        'lr_coefficient': np.abs(lr_model.coef_)
    }).sort_values('lr_coefficient', ascending=False)
    
    # XGBoost feature importance
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'xgb_importance': xgb_model.feature_importances_
    }).sort_values('xgb_importance', ascending=False)
    
    # Merge and get top 10
    importance_df = lr_importance.merge(xgb_importance, on='feature')
    importance_df['lr_coefficient_norm'] = importance_df['lr_coefficient'] / importance_df['lr_coefficient'].max()
    importance_df['xgb_importance_norm'] = importance_df['xgb_importance'] / importance_df['xgb_importance'].max()
    
    top_features = importance_df.nlargest(10, 'xgb_importance')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(top_features))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, top_features['lr_coefficient_norm'], width, 
                    label='Linear Regression', alpha=0.8, color='steelblue')
    bars2 = ax.barh(x + width/2, top_features['xgb_importance_norm'], width, 
                    label='XGBoost', alpha=0.8, color='darkorange')
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Normalized Importance', fontweight='bold', fontsize=12)
    ax.set_title('Top 10 Feature Importance Comparison', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature importance comparison saved")

def create_error_by_range_analysis(y_true, lr_pred, ridge_pred, xgb_pred, output_dir):
    """Analyze prediction error by loan amount range."""
    
    # Create ranges
    ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
    range_labels = ['$0-500', '$500-1000', '$1000-1500', '$1500-2000', '$2000-3000']
    
    lr_errors = []
    ridge_errors = []
    xgb_errors = []
    
    for low, high in ranges:
        mask = (y_true >= low) & (y_true < high)
        
        lr_errors.append(mean_absolute_error(y_true[mask], lr_pred[mask]))
        ridge_errors.append(mean_absolute_error(y_true[mask], ridge_pred[mask]))
        xgb_errors.append(mean_absolute_error(y_true[mask], xgb_pred[mask]))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(range_labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, lr_errors, width, label='Linear Regression', 
                   alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x, ridge_errors, width, label='Ridge Regression', 
                   alpha=0.8, color='seagreen', edgecolor='black')
    bars3 = ax.bar(x + width, xgb_errors, width, label='XGBoost', 
                   alpha=0.8, color='darkorange', edgecolor='black')
    
    ax.set_ylabel('Mean Absolute Error ($)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Loan Amount Range', fontweight='bold', fontsize=12)
    ax.set_title('Prediction Error by Loan Amount Range', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_by_range.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Error by range analysis saved")

if __name__ == "__main__":
    create_comprehensive_visualizations()
