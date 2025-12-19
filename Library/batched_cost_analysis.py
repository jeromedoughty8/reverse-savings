
"""
Batched Cost Analysis: FP/FN Weighted Cost Analysis in Batches of 100

This module analyzes False Positives (FP) and False Negatives (FN) in batches,
applying the business cost formula: FP + FN*10

For each batch:
- If (FP + FN*10) >= 2, track the batch
- Calculate opportunity loss: (sum - frequency) * 1000
- Track frequency percentage across all batches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def analyze_batched_costs(y_true, y_pred, batch_size=100, threshold=2, fn_weight=10, fp_weight=1):
    """
    Analyze FP/FN costs in batches with threshold logic.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        batch_size: Size of each batch (default 100)
        threshold: Threshold for counting batch (default 2)
        fn_weight: Weight for False Negatives (default 10)
        fp_weight: Weight for False Positives (default 1)
    
    Returns:
        results_df: DataFrame with batch-level analysis
        summary: Dictionary with aggregated metrics
    """
    
    n_samples = len(y_true)
    n_batches = n_samples // batch_size
    
    # Trim to fit exact batches
    y_true_batched = y_true[:n_batches * batch_size]
    y_pred_batched = y_pred[:n_batches * batch_size]
    
    batch_results = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Get batch data
        y_true_batch = y_true_batched[start_idx:end_idx]
        y_pred_batch = y_pred_batched[start_idx:end_idx]
        
        # Calculate confusion matrix for this batch
        cm = confusion_matrix(y_true_batch, y_pred_batch, labels=[1, 0])
        
        # Extract FP and FN
        fn = cm[0, 1]  # False Negatives (missed defaults)
        fp = cm[1, 0]  # False Positives (rejected good customers)
        
        # Calculate weighted cost: FP*1 + FN*10
        weighted_cost = (fp * fp_weight) + (fn * fn_weight)
        
        # Check if batch meets threshold
        meets_threshold = weighted_cost >= threshold
        
        batch_results.append({
            'batch_id': i + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'fp': fp,
            'fn': fn,
            'weighted_cost': weighted_cost,
            'meets_threshold': meets_threshold
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(batch_results)
    
    # Calculate summary metrics
    batches_meeting_threshold = results_df[results_df['meets_threshold']]
    frequency = len(batches_meeting_threshold)
    total_batches = len(results_df)
    
    if frequency > 0:
        sum_of_values = batches_meeting_threshold['weighted_cost'].sum()
        opportunity_loss = (sum_of_values - frequency) * 1000
    else:
        sum_of_values = 0
        opportunity_loss = 0
    
    frequency_percentage = (frequency / total_batches) * 100
    
    summary = {
        'total_batches': total_batches,
        'batch_size': batch_size,
        'threshold': threshold,
        'fn_weight': fn_weight,
        'fp_weight': fp_weight,
        'batches_meeting_threshold': frequency,
        'frequency_percentage': frequency_percentage,
        'sum_of_weighted_costs': sum_of_values,
        'opportunity_loss': opportunity_loss,
        'avg_weighted_cost_all_batches': results_df['weighted_cost'].mean(),
        'avg_weighted_cost_threshold_batches': batches_meeting_threshold['weighted_cost'].mean() if frequency > 0 else 0,
        'total_fp': results_df['fp'].sum(),
        'total_fn': results_df['fn'].sum()
    }
    
    return results_df, summary


def save_batched_analysis_table(results_df, summary, output_dir='data/batched_cost_analysis'):
    """
    Save batch analysis results to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed batch results
    batch_file = os.path.join(output_dir, 'batch_level_results.csv')
    results_df.to_csv(batch_file, index=False)
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, 'summary_metrics.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # Save batches meeting threshold only
    threshold_batches = results_df[results_df['meets_threshold']]
    threshold_file = os.path.join(output_dir, 'batches_meeting_threshold.csv')
    threshold_batches.to_csv(threshold_file, index=False)
    
    return batch_file, summary_file, threshold_file


def plot_batched_analysis(results_df, summary, output_dir='data/batched_cost_analysis'):
    """
    Create comprehensive visualizations for batched cost analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Weighted Cost per Batch
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['red' if x else 'lightgray' for x in results_df['meets_threshold']]
    ax1.bar(results_df['batch_id'], results_df['weighted_cost'], color=colors, alpha=0.7)
    ax1.axhline(y=summary['threshold'], color='black', linestyle='--', linewidth=2, label=f'Threshold = {summary["threshold"]}')
    ax1.set_xlabel('Batch ID', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Weighted Cost (FP + FN×10)', fontweight='bold', fontsize=12)
    ax1.set_title(f'Weighted Cost per Batch (Batch Size = {summary["batch_size"]})', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: FP vs FN per Batch
    ax2 = fig.add_subplot(gs[1, 0])
    threshold_batches = results_df[results_df['meets_threshold']]
    if len(threshold_batches) > 0:
        ax2.scatter(threshold_batches['fp'], threshold_batches['fn'], 
                   s=100, alpha=0.6, c='red', label='Meets Threshold')
    non_threshold = results_df[~results_df['meets_threshold']]
    if len(non_threshold) > 0:
        ax2.scatter(non_threshold['fp'], non_threshold['fn'], 
                   s=50, alpha=0.3, c='gray', label='Below Threshold')
    ax2.set_xlabel('False Positives (FP)', fontweight='bold')
    ax2.set_ylabel('False Negatives (FN)', fontweight='bold')
    ax2.set_title('FP vs FN Distribution by Batch', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of Weighted Costs
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(results_df['weighted_cost'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=summary['threshold'], color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {summary["threshold"]}')
    ax3.set_xlabel('Weighted Cost', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Distribution of Weighted Costs Across Batches', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Metrics
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    summary_text = f"""
    BATCHED COST ANALYSIS SUMMARY
    ═══════════════════════════════════════════════
    
    Batch Configuration:
    • Total Batches: {summary['total_batches']:,}
    • Batch Size: {summary['batch_size']}
    • Threshold: {summary['threshold']}
    • FN Weight: {summary['fn_weight']}x
    • FP Weight: {summary['fp_weight']}x
    
    Threshold Analysis:
    • Batches Meeting Threshold (≥{summary['threshold']}): {summary['batches_meeting_threshold']:,}
    • Frequency Percentage: {summary['frequency_percentage']:.2f}%
    • Sum of Weighted Costs (≥{summary['threshold']}): {summary['sum_of_weighted_costs']:.2f}
    
    Financial Impact:
    • Opportunity Loss: ${summary['opportunity_loss']:,.2f}
      Formula: (Sum - Frequency) × 1,000
      = ({summary['sum_of_weighted_costs']:.2f} - {summary['batches_meeting_threshold']}) × 1,000
    
    Overall Metrics:
    • Total FP (All Batches): {summary['total_fp']:,}
    • Total FN (All Batches): {summary['total_fn']:,}
    • Avg Weighted Cost (All): {summary['avg_weighted_cost_all_batches']:.2f}
    • Avg Weighted Cost (≥{summary['threshold']}): {summary['avg_weighted_cost_threshold_batches']:.2f}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 5: Cumulative Analysis
    ax5 = fig.add_subplot(gs[2, 1])
    cumulative_cost = results_df['weighted_cost'].cumsum()
    cumulative_threshold = results_df['meets_threshold'].cumsum()
    
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(results_df['batch_id'], cumulative_cost, 'b-', linewidth=2, label='Cumulative Weighted Cost')
    line2 = ax5_twin.plot(results_df['batch_id'], cumulative_threshold, 'r-', linewidth=2, label='Cumulative Threshold Count')
    
    ax5.set_xlabel('Batch ID', fontweight='bold')
    ax5.set_ylabel('Cumulative Weighted Cost', fontweight='bold', color='b')
    ax5_twin.set_ylabel('Cumulative Batches ≥ Threshold', fontweight='bold', color='r')
    ax5.set_title('Cumulative Analysis', fontweight='bold')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5_twin.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Save figure
    output_file = os.path.join(output_dir, 'batched_cost_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def print_batched_analysis_report(summary):
    """
    Print formatted report of batched cost analysis.
    """
    print("\n" + "="*80)
    print("BATCHED COST ANALYSIS REPORT")
    print("="*80)
    print("\nBATCH CONFIGURATION:")
    print("-"*80)
    print(f"Total Batches:              {summary['total_batches']:,}")
    print(f"Batch Size:                 {summary['batch_size']}")
    print(f"Cost Threshold:             {summary['threshold']}")
    print(f"FN Weight (Missed Default): {summary['fn_weight']}x")
    print(f"FP Weight (Rejected Good):  {summary['fp_weight']}x")
    
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS:")
    print("-"*80)
    print(f"Batches Meeting Threshold:  {summary['batches_meeting_threshold']:,} / {summary['total_batches']:,}")
    print(f"Frequency Percentage:       {summary['frequency_percentage']:.2f}%")
    print(f"Sum of Weighted Costs:      {summary['sum_of_weighted_costs']:.2f}")
    
    print("\n" + "="*80)
    print("FINANCIAL IMPACT:")
    print("-"*80)
    print(f"Opportunity Loss Formula:   (Sum - Frequency) × 1,000")
    print(f"                           ({summary['sum_of_weighted_costs']:.2f} - {summary['batches_meeting_threshold']}) × 1,000")
    print(f"Opportunity Loss:           ${summary['opportunity_loss']:,.2f}")
    
    print("\n" + "="*80)
    print("AGGREGATE METRICS:")
    print("-"*80)
    print(f"Total False Positives:      {summary['total_fp']:,}")
    print(f"Total False Negatives:      {summary['total_fn']:,}")
    print(f"Avg Cost (All Batches):     {summary['avg_weighted_cost_all_batches']:.2f}")
    print(f"Avg Cost (≥{summary['threshold']} Only):      {summary['avg_weighted_cost_threshold_batches']:.2f}")
    print("="*80 + "\n")


def run_batched_cost_analysis(y_true, y_pred, batch_size=100, threshold=2, 
                               fn_weight=10, fp_weight=1, output_dir='data/batched_cost_analysis'):
    """
    Complete workflow for batched cost analysis.
    """
    print("\n" + "="*80)
    print("RUNNING BATCHED COST ANALYSIS")
    print("="*80)
    
    # Run analysis
    results_df, summary = analyze_batched_costs(
        y_true, y_pred, batch_size, threshold, fn_weight, fp_weight
    )
    
    # Save tables
    batch_file, summary_file, threshold_file = save_batched_analysis_table(
        results_df, summary, output_dir
    )
    
    # Create visualizations
    plot_file = plot_batched_analysis(results_df, summary, output_dir)
    
    # Print report
    print_batched_analysis_report(summary)
    
    print(f"✓ Batch-level results saved to: {batch_file}")
    print(f"✓ Summary metrics saved to: {summary_file}")
    print(f"✓ Threshold batches saved to: {threshold_file}")
    print(f"✓ Visualizations saved to: {plot_file}")
    
    return results_df, summary
