
"""
Threshold Optimization for Credit Risk Models
Finds optimal decision threshold to balance precision and recall
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score
import os


def find_optimal_threshold(y_true, y_pred_proba, min_recall=0.70, output_dir='data/threshold_optimization'):
    """
    Find optimal classification threshold to achieve target recall.
    
    Args:
        y_true: Actual labels
        y_pred_proba: Predicted probabilities
        min_recall: Minimum acceptable recall (default 70%)
        output_dir: Directory to save plots
        
    Returns:
        optimal_threshold: Best threshold that meets recall requirement
        metrics: Performance metrics at that threshold
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION FOR RECALL")
    print("=" * 70)
    print(f"Target: Recall ≥ {min_recall:.0%}")
    print()
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Find thresholds that meet recall requirement
    valid_indices = np.where(recalls[:-1] >= min_recall)[0]
    
    if len(valid_indices) == 0:
        print(f"⚠ Warning: Cannot achieve {min_recall:.0%} recall")
        print("  Using threshold that maximizes F1-score instead")
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
        best_idx = np.argmax(f1_scores)
    else:
        # Among valid thresholds, choose one with best precision
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
    
    optimal_threshold = thresholds[best_idx]
    optimal_precision = precisions[best_idx]
    optimal_recall = recalls[best_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'precision': precision_score(y_true, y_pred_optimal),
        'recall': recall_score(y_true, y_pred_optimal),
        'f1_score': f1_score(y_true, y_pred_optimal)
    }
    
    print(f"OPTIMAL THRESHOLD: {optimal_threshold:.3f}")
    print("-" * 70)
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1-Score:  {metrics['f1_score']:.2%}")
    print()
    
    # Compare with default threshold (0.5)
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    default_recall = recall_score(y_true, y_pred_default)
    default_precision = precision_score(y_true, y_pred_default)
    
    print(f"COMPARISON WITH DEFAULT (0.5):")
    print("-" * 70)
    print(f"Default Precision: {default_precision:.2%}")
    print(f"Default Recall:    {default_recall:.2%}")
    print(f"Improvement:       +{(metrics['recall'] - default_recall)*100:.1f} percentage points in recall")
    print()
    
    # Plot precision-recall trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precision-Recall curve
    ax1.plot(recalls[:-1], precisions[:-1], 'b-', linewidth=2, label='PR Curve')
    ax1.plot(optimal_recall, optimal_precision, 'ro', markersize=10, 
             label=f'Optimal (threshold={optimal_threshold:.3f})')
    ax1.axhline(y=min_recall, color='g', linestyle='--', label=f'Target Recall = {min_recall:.0%}')
    ax1.set_xlabel('Recall', fontweight='bold')
    ax1.set_ylabel('Precision', fontweight='bold')
    ax1.set_title('Precision-Recall Trade-off', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Threshold vs Metrics
    ax2.plot(thresholds, precisions[:-1], 'b-', label='Precision', linewidth=2)
    ax2.plot(thresholds, recalls[:-1], 'r-', label='Recall', linewidth=2)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    ax2.plot(thresholds, f1_scores, 'g-', label='F1-Score', linewidth=2)
    ax2.axvline(x=optimal_threshold, color='k', linestyle='--', 
                label=f'Optimal = {optimal_threshold:.3f}')
    ax2.set_xlabel('Threshold', fontweight='bold')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Metrics vs Threshold', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/threshold_optimization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_dir}/threshold_optimization.png")
    
    # Save results
    results_df = pd.DataFrame([{
        'Optimal_Threshold': optimal_threshold,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1_Score': metrics['f1_score'],
        'Default_Threshold': 0.5,
        'Default_Recall': default_recall,
        'Recall_Improvement': metrics['recall'] - default_recall
    }])
    
    results_df.to_csv(f'{output_dir}/threshold_results.csv', index=False)
    print(f"✓ Results saved to: {output_dir}/threshold_results.csv")
    print()
    print("=" * 70)
    
    return optimal_threshold, metrics


if __name__ == "__main__":
    # Example usage
    import joblib
    import numpy as np
    
    # Load model and data
    data = np.load('data/models_output/processed_data.npz', allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_class_test']
    
    model = joblib.load('models/xgboost_primary.pkl')
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    optimal_threshold, metrics = find_optimal_threshold(y_test, y_pred_proba, min_recall=0.70)
    
    print(f"\nRecommendation: Use threshold = {optimal_threshold:.3f} in production")
