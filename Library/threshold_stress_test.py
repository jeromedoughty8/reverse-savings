"""
Threshold Stress Test Analysis
Compares profitability and risk metrics across different classification thresholds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


class ThresholdStressTest:
    """
    Analyze business impact across multiple threshold scenarios.
    Compares profit, costs (FN/FP), and Expected Credit Loss (ECL).
    """

    def __init__(self, y_true, y_pred_proba, avg_loan_amount=700,
                 subscription_ltv=106.92, cost_per_default=1000):
        """
        Initialize stress test with predictions and business parameters.

        Args:
            y_true: Actual labels (0=Repays, 1=Default)
            y_pred_proba: Predicted default probabilities
            avg_loan_amount: Average loan size
            subscription_ltv: Lifetime value per customer (~6 months × $17.82)
            cost_per_default: Direct loss per defaulted loan
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.avg_loan_amount = avg_loan_amount
        self.subscription_ltv = subscription_ltv
        self.cost_per_default = cost_per_default

    def analyze_threshold(self, threshold):
        """
        Calculate all metrics for a specific threshold.

        Returns:
            Dictionary with financial and risk metrics
        """
        # Make predictions at this threshold
        y_pred = (self.y_pred_proba >= threshold).astype(int)

        # Confusion matrix
        cm = confusion_matrix(self.y_true, y_pred, labels=[1, 0])
        tp = cm[0, 0]  # Correctly rejected defaults
        fn = cm[0, 1]  # Missed defaults (approved bad customers)
        fp = cm[1, 0]  # Rejected good customers
        tn = cm[1, 1]  # Correctly approved good customers

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Customer decisions
        approved_customers = tn + fn
        rejected_customers = tp + fp
        total_customers = len(self.y_true)

        # Revenue calculations
        # ALL customers pay subscription, not just those who borrow
        subscription_revenue = total_customers * self.subscription_ltv

        # Cost calculations
        fn_cost = fn * self.cost_per_default  # Direct losses from missed defaults
        # FP cost = opportunity cost of not providing loans to good customers
        # Even though they pay subscription, we lose customer satisfaction and potential upsells
        fp_cost = fp * 100  # $100 opportunity cost per rejected good customer
        total_cost = fn_cost + fp_cost

        # Profit
        gross_profit = subscription_revenue
        net_profit = gross_profit - total_cost
        profit_margin = (net_profit / gross_profit * 100) if gross_profit > 0 else 0

        # Expected Credit Loss (ECL)
        # ECL = Default Rate × Exposure × Loss Given Default
        default_rate = fn / approved_customers if approved_customers > 0 else 0
        total_exposure = approved_customers * self.avg_loan_amount
        lgd = 1.0  # Loss Given Default = 100% (can't recover defaulted loans)
        ecl = default_rate * total_exposure * lgd

        # Risk metrics
        approval_rate = approved_customers / total_customers * 100
        fp_fn_ratio = fp / fn if fn > 0 else float('inf')  # FP/FN: >1 means rejecting too many good customers
        # Cost ratio using person counts: (FP × 0.1) / FN
        # 0.1 = $100 FP cost / $1000 FN cost
        # Ratio = 1.0 means 10 FPs = 1 FN in cost (break even)
        # Ratio > 1.0 means rejecting too many good customers (losing money)
        cost_fp_fn_ratio = (fp * 0.1) / fn if fn > 0 else float('inf')

        return {
            'threshold': threshold,
            'approved_customers': approved_customers,
            'rejected_customers': rejected_customers,
            'approval_rate_pct': approval_rate,

            # Confusion matrix components
            'true_positives': tp,
            'false_negatives': fn,
            'false_positives': fp,
            'true_negatives': tn,

            # Model performance metrics
            'precision': precision,
            'recall': recall,

            # Revenue & Profit
            'subscription_revenue': subscription_revenue,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'profit_margin_pct': profit_margin,

            # Costs
            'fn_cost': fn_cost,
            'fp_cost': fp_cost,
            'total_cost': total_cost,

            # Ratios (FP/FN: >1 = rejecting too many good customers)
            'fp_fn_ratio': fp_fn_ratio,
            'cost_fp_fn_ratio': cost_fp_fn_ratio,

            # Risk metrics
            'default_rate_pct': default_rate * 100,
            'total_exposure': total_exposure,
            'expected_credit_loss': ecl,
            'ecl_rate_pct': (ecl / total_exposure * 100) if total_exposure > 0 else 0
        }

    def run_stress_test(self, thresholds=None):
        """
        Run stress test across multiple thresholds.

        Args:
            thresholds: List of thresholds to test (default: 5% to 100% in 5% increments)

        Returns:
            DataFrame with results for all thresholds
        """
        if thresholds is None:
            # Default: 5% to 100% in 5% increments
            thresholds = [i/100.0 for i in range(5, 105, 5)]

        results = []

        for threshold in thresholds:
            metrics = self.analyze_threshold(threshold)
            results.append(metrics)

        df = pd.DataFrame(results)
        return df

    def generate_comparison_table(self, df, output_dir='data/stress_test'):
        """
        Generate formatted comparison table.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create display table with formatted values
        display_df = pd.DataFrame({
            'Threshold': [f"{t:.0%}" for t in df['threshold']],
            'Approval Rate': [f"{r:.1f}%" for r in df['approval_rate_pct']],
            'Net Profit': [f"${p:,.0f}" for p in df['net_profit']],
            'Profit Margin': [f"{m:.1f}%" for m in df['profit_margin_pct']],
            'FN Cost': [f"${c:,.0f}" for c in df['fn_cost']],
            'FP Cost': [f"${c:,.0f}" for c in df['fp_cost']],
            'Total Cost': [f"${c:,.0f}" for c in df['total_cost']],
            'FP/FN Ratio': [f"{r:.2f}" if r != float('inf') else '∞' for r in df['fp_fn_ratio']],
            'Cost Ratio': [f"{r:.2f}" if r != float('inf') else '∞' for r in df['cost_fp_fn_ratio']],
            'Default Rate': [f"{r:.2f}%" for r in df['default_rate_pct']],
            'ECL': [f"${e:,.0f}" for e in df['expected_credit_loss']],
            'ECL Rate': [f"{r:.2f}%" for r in df['ecl_rate_pct']]
        })

        # Save to CSV
        display_df.to_csv(f'{output_dir}/threshold_comparison_table.csv', index=False)
        df.to_csv(f'{output_dir}/threshold_comparison_raw.csv', index=False)

        # Print formatted table to console
        print("\n" + "=" * 140)
        print("THRESHOLD COMPARISON TABLE (DETAILED)")
        print("=" * 140)
        print(display_df.to_string(index=False))
        print("=" * 140)

        return display_df

    def plot_heatmap_comparison(self, df, output_dir='data/stress_test'):
        """
        Create heatmap visualization with actual values and quality-based colors.
        Colors represent good (green) vs bad (red) for each metric.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data with ACTUAL values - clearer formatting for readability
        heatmap_data = pd.DataFrame({
            'Threshold': [f"{t:.0%}" for t in df['threshold']],
            'Net Profit': [f"${p/1e6:.1f}M" for p in df['net_profit']],
            'Recall %': [f"{r*100:.0f}%" for r in df['recall']],
            'Precision %': [f"{p*100:.0f}%" for p in df['precision']],
            'Approval %': [f"{a:.0f}%" for a in df['approval_rate_pct']],
            'FN Cost': [f"${c/1e3:.0f}K" for c in df['fn_cost']],
            'FP Cost': [f"${c/1e3:.0f}K" for c in df['fp_cost']],
            'Cost Ratio\n(FP×0.1/FN)': [f"{r:.2f}" if r != float('inf') else '∞' for r in df['cost_fp_fn_ratio']],
        })

        # Create quality scores for coloring (0-100 scale where higher = better)
        quality_scores = pd.DataFrame({
            'Threshold': df['threshold'],
            'Net Profit': ((df['net_profit'] - df['net_profit'].min()) / 
                          (df['net_profit'].max() - df['net_profit'].min()) * 100),
            'Recall %': df['recall'] * 100,
            'Precision %': df['precision'] * 100,
            'Approval %': df['approval_rate_pct'],
            'FN Cost': (100 - ((df['fn_cost'] - df['fn_cost'].min()) / 
                              (df['fn_cost'].max() - df['fn_cost'].min()) * 100)),  # Inverted: lower cost is better
            'FP Cost': (100 - ((df['fp_cost'] - df['fp_cost'].min()) / 
                              (df['fp_cost'].max() - df['fp_cost'].min()) * 100)),  # Inverted: lower cost is better
            # Cost Ratio: (FP × 0.1) / FN
            # Ratio ≥ 1 = Rejecting 10+ good customers per bad customer (TOO CONSERVATIVE, BAD)
            # Ratio < 1 = Efficient balance (GOOD)
            'Cost Ratio\n(FP×0.1/FN)': [100 - min(r * 50, 100) if r != float('inf') else 0 
                                       for r in df['cost_fp_fn_ratio']],  # Inverted: lower ratio is better
        })

        # Create heatmap with actual values displayed but colors based on quality
        fig, ax = plt.subplots(figsize=(22, 12))

        # Transpose for better visualization (thresholds as columns)
        display_matrix = heatmap_data.set_index('Threshold').T
        quality_matrix = quality_scores.set_index('Threshold').T

        # Create custom colormap: Red (bad) -> Yellow (ok) -> Green (good)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#8B0000', '#DC143C', '#FF6347', '#FFA500', '#FFD700', '#90EE90', '#32CD32', '#006400']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('quality', colors_list, N=n_bins)

        # Create heatmap with quality scores for colors but actual values for annotations
        sns.heatmap(quality_matrix, annot=display_matrix, fmt='', cmap=cmap,
                    cbar_kws={'label': 'Quality Score (Dark Green = Excellent, Red = Poor)', 'shrink': 0.7},
                    linewidths=2.0, linecolor='white', ax=ax, vmin=0, vmax=100,
                    annot_kws={'fontsize': 13, 'fontweight': 'bold', 'ha': 'center'})

        ax.set_title('Threshold Performance Heatmap\n(Values shown with quality-based colors)',
                     fontweight='bold', fontsize=18, pad=25)
        ax.set_xlabel('Classification Threshold', fontweight='bold', fontsize=15)
        ax.set_ylabel('Performance Metric', fontweight='bold', fontsize=15)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=13)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)

        # Add legend explanation
        legend_text = (
            "COST RATIO EXPLANATION:\n"
            "Cost Ratio = (FP × 0.1) ÷ FN\n"
            "Net Loss = (Ratio - 1) × $1,000\n\n"
            "• Ratio = 1.0 → $0 net loss\n"
            "  (10 good rejected per 1 bad caught)\n"
            "  → BREAK EVEN\n\n"
            "• Ratio = 1.1 → $100 net loss\n"
            "  → Slightly over break-even\n\n"
            "• Ratio = 2.0 → $1,000 net loss\n"
            "  (2× opp cost vs direct loss)\n"
            "  → TOO CONSERVATIVE\n\n"
            "• Ratio = 3.0 → $2,000 net loss\n"
            "  → VERY BAD\n\n"
            "• Ratio < 1.0 → Good balance\n"
            "• Ratio ≥ 2.0 → Adjust threshold\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "COLOR GUIDE:\n"
            "• Dark Green = Excellent\n"
            "• Light Green = Good\n"
            "• Yellow = Acceptable\n"
            "• Orange = Poor\n"
            "• Red = Bad\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "METRICS:\n"
            "• Net Profit: Higher = Better\n"
            "• Recall %: Higher = Better\n"
            "• Precision %: Higher = Better\n"
            "• FN/FP Costs: Lower = Better\n"
            "• Cost Ratio: <1.0 ideal"
        )
        ax.text(1.02, 0.5, legend_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1.5))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/threshold_heatmap_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create focused Net Profit vs ECL heatmap with actual values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        from matplotlib.colors import LinearSegmentedColormap

        # Net Profit heatmap - display in millions
        profit_display = df[['threshold', 'net_profit']].copy()
        profit_display['net_profit_millions'] = profit_display['net_profit'] / 1e6
        profit_display_matrix = profit_display[['threshold', 'net_profit_millions']].set_index('threshold').T
        profit_display_matrix.index = ['Net Profit ($M)']
        profit_display_matrix.columns = [f"{t:.0%}" for t in profit_display_matrix.columns]

        # Quality scores for profit (normalized 0-100)
        profit_quality = ((df['net_profit'] - df['net_profit'].min()) / 
                         (df['net_profit'].max() - df['net_profit'].min()) * 100)
        profit_quality_matrix = pd.DataFrame([profit_quality.values], 
                                            columns=profit_display_matrix.columns,
                                            index=['Net Profit ($M)'])

        # Custom colormap for profit
        profit_cmap = LinearSegmentedColormap.from_list('profit', 
                                                        ['darkred', 'red', 'orange', 'yellow', 
                                                         'lightgreen', 'green', 'darkgreen'], N=100)

        # Create annotations with actual values
        profit_annot = profit_display_matrix.map(lambda x: f"${x:.2f}M")

        sns.heatmap(profit_quality_matrix, annot=profit_annot, fmt='', cmap=profit_cmap,
                    cbar_kws={'label': 'Quality (Green = High Profit, Red = Low Profit)', 'shrink': 0.8},
                    linewidths=1.0, linecolor='white', ax=ax1, vmin=0, vmax=100,
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Net Profit by Threshold', fontweight='bold', fontsize=13, pad=10)
        ax1.set_xlabel('Classification Threshold', fontweight='bold', fontsize=12)
        ax1.set_ylabel('')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=11)

        # ECL heatmap - display in thousands
        ecl_display = df[['threshold', 'expected_credit_loss']].copy()
        ecl_display['ecl_thousands'] = ecl_display['expected_credit_loss'] / 1e3
        ecl_display_matrix = ecl_display[['threshold', 'ecl_thousands']].set_index('threshold').T
        ecl_display_matrix.index = ['Expected Credit Loss ($K)']
        ecl_display_matrix.columns = [f"{t:.0%}" for t in ecl_display_matrix.columns]

        # Quality scores for ECL (inverted: lower ECL = better = higher score)
        ecl_quality = (100 - ((df['expected_credit_loss'] - df['expected_credit_loss'].min()) / 
                             (df['expected_credit_loss'].max() - df['expected_credit_loss'].min()) * 100))
        ecl_quality_matrix = pd.DataFrame([ecl_quality.values], 
                                         columns=ecl_display_matrix.columns,
                                         index=['Expected Credit Loss ($K)'])

        # Custom colormap for ECL (same as profit - green is good/low risk)
        ecl_cmap = LinearSegmentedColormap.from_list('ecl', 
                                                     ['darkred', 'red', 'orange', 'yellow', 
                                                      'lightgreen', 'green', 'darkgreen'], N=100)

        # Create annotations with actual values
        ecl_annot = ecl_display_matrix.map(lambda x: f"${x:.0f}K")

        sns.heatmap(ecl_quality_matrix, annot=ecl_annot, fmt='', cmap=ecl_cmap,
                    cbar_kws={'label': 'Quality (Green = Low Risk, Red = High Risk)', 'shrink': 0.8},
                    linewidths=1.0, linecolor='white', ax=ax2, vmin=0, vmax=100,
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Expected Credit Loss by Threshold', fontweight='bold', fontsize=13, pad=10)
        ax2.set_xlabel('Classification Threshold', fontweight='bold', fontsize=12)
        ax2.set_ylabel('')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=11)

        plt.suptitle('Net Profit vs. Credit Risk: Threshold Comparison\n(Actual Values with Quality-Based Colors)',
                     fontweight='bold', fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/threshold_profit_vs_ecl_heatmap.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Heatmap visualizations saved to {output_dir}/")

    def plot_stress_test_results(self, df, output_dir='data/stress_test'):
        """
        Create comprehensive visualization with dual y-axes.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Primary y-axis: Profit metrics
        color_profit = 'green'
        ax1.set_xlabel('Classification Threshold', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Profit ($)', color=color_profit, fontweight='bold', fontsize=12)

        # Plot net profit
        line1 = ax1.plot(df['threshold'], df['net_profit'],
                        color=color_profit, linewidth=3, marker='o', markersize=8,
                        label='Net Profit', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color_profit)
        ax1.grid(True, alpha=0.3)

        # Format x-axis as percentages
        ax1.set_xticks(df['threshold'])
        ax1.set_xticklabels([f"{t:.0%}" for t in df['threshold']])

        # Secondary y-axis: ECL
        ax2 = ax1.twinx()
        color_ecl = 'red'
        ax2.set_ylabel('Expected Credit Loss (ECL) ($)', color=color_ecl,
                      fontweight='bold', fontsize=12)

        line2 = ax2.plot(df['threshold'], df['expected_credit_loss'],
                        color=color_ecl, linewidth=3, marker='s', markersize=8,
                        label='Expected Credit Loss', alpha=0.8, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color_ecl)

        # Add additional profit-related lines on primary axis
        line3 = ax1.plot(df['threshold'], df['fn_cost'],
                        color='orange', linewidth=2, marker='^', markersize=6,
                        label='FN Cost (Direct Loss)', alpha=0.7, linestyle='-.')

        line4 = ax1.plot(df['threshold'], df['fp_cost'],
                        color='blue', linewidth=2, marker='v', markersize=6,
                        label='FP Cost (Opportunity)', alpha=0.7, linestyle=':')

        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', framealpha=0.9, fontsize=10)

        # Title
        plt.title('Threshold Stress Test: Profit vs. Credit Risk',
                 fontweight='bold', fontsize=14, pad=20)

        # Add annotations for key thresholds
        for idx, row in df.iterrows():
            if row['threshold'] in [0.05, 0.25]:
                ax1.annotate(f"${row['net_profit']:,.0f}",
                           xy=(row['threshold'], row['net_profit']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/threshold_stress_test_dual_axis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create second visualization: Multi-metric comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Threshold Stress Test: Comprehensive Metrics',
                    fontweight='bold', fontsize=16)

        # Plot 1: Profit metrics
        axes[0, 0].plot(df['threshold'], df['net_profit'],
                       'g-o', linewidth=2, label='Net Profit', markersize=8)
        axes[0, 0].plot(df['threshold'], df['gross_profit'],
                       'b--s', linewidth=2, label='Gross Profit', markersize=6, alpha=0.7)
        axes[0, 0].set_xlabel('Threshold', fontweight='bold')
        axes[0, 0].set_ylabel('Profit ($)', fontweight='bold')
        axes[0, 0].set_title('Profitability by Threshold', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(df['threshold'])
        axes[0, 0].set_xticklabels([f"{t:.0%}" for t in df['threshold']])

        # Plot 2: Cost breakdown
        axes[0, 1].plot(df['threshold'], df['fn_cost'],
                       'r-o', linewidth=2, label='FN Cost', markersize=8)
        axes[0, 1].plot(df['threshold'], df['fp_cost'],
                       'b-s', linewidth=2, label='FP Cost', markersize=8)
        axes[0, 1].plot(df['threshold'], df['total_cost'],
                       'k--^', linewidth=2, label='Total Cost', markersize=6, alpha=0.7)
        axes[0, 1].set_xlabel('Threshold', fontweight='bold')
        axes[0, 1].set_ylabel('Cost ($)', fontweight='bold')
        axes[0, 1].set_title('Cost Structure by Threshold', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(df['threshold'])
        axes[0, 1].set_xticklabels([f"{t:.0%}" for t in df['threshold']])

        # Plot 3: Ratios
        axes[1, 0].plot(df['threshold'], df['fp_fn_ratio'],
                       color='purple', linewidth=2, marker='o', label='FP/FN Ratio', markersize=8)
        axes[1, 0].axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Balance Point (FP=FN)')
        axes[1, 0].set_xlabel('Threshold', fontweight='bold')
        axes[1, 0].set_ylabel('FP/FN Ratio', fontweight='bold')
        axes[1, 0].set_title('FP/FN Ratio by Threshold\n(>1 = Too Conservative)', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(df['threshold'])
        axes[1, 0].set_xticklabels([f"{t:.0%}" for t in df['threshold']])

        # Plot 4: ECL and Default Rate
        ax_left = axes[1, 1]
        ax_right = ax_left.twinx()

        line1 = ax_left.plot(df['threshold'], df['expected_credit_loss'],
                            color='r', linewidth=2, marker='o', label='ECL ($)', markersize=8)
        ax_left.set_xlabel('Threshold', fontweight='bold')
        ax_left.set_ylabel('Expected Credit Loss ($)', color='r', fontweight='bold')
        ax_left.tick_params(axis='y', labelcolor='r')

        line2 = ax_right.plot(df['threshold'], df['default_rate_pct'],
                             color='b', linewidth=2, marker='s', label='Default Rate (%)', markersize=8)
        ax_right.set_ylabel('Default Rate (%)', color='b', fontweight='bold')
        ax_right.tick_params(axis='y', labelcolor='b')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper left')
        ax_left.set_title('Credit Risk Metrics by Threshold', fontweight='bold')
        ax_left.grid(True, alpha=0.3)
        ax_left.set_xticks(df['threshold'])
        ax_left.set_xticklabels([f"{t:.0%}" for t in df['threshold']])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/threshold_stress_test_comprehensive.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def print_summary_report(self, df):
        """
        Print formatted summary report.
        """
        print("\n" + "=" * 100)
        print("THRESHOLD STRESS TEST ANALYSIS")
        print("=" * 100)
        print(f"\nBusiness Parameters:")
        print(f"  • Average Loan Amount: ${self.avg_loan_amount:,.2f}")
        print(f"  • Subscription LTV: ${self.subscription_ltv:,.2f}")
        print(f"  • Cost per Default: ${self.cost_per_default:,.2f}")
        print(f"  • Total Test Samples: {len(self.y_true):,}")
        print()

        print("=" * 100)
        print("THRESHOLD COMPARISON TABLE")
        print("=" * 100)
        print(f"{'Threshold':<12} {'Approval':<10} {'Net Profit':<15} {'Margin':<10} "
              f"{'FN Cost':<15} {'FP Cost':<15} {'ECL':<15} {'FP/FN':<8}")
        print("-" * 100)

        for _, row in df.iterrows():
            fp_fn_str = f"{row['fp_fn_ratio']:.2f}" if row['fp_fn_ratio'] != float('inf') else '∞'
            print(f"{row['threshold']:.0%}          "
                  f"{row['approval_rate_pct']:>6.1f}%    "
                  f"${row['net_profit']:>12,.0f}   "
                  f"{row['profit_margin_pct']:>6.1f}%    "
                  f"${row['fn_cost']:>12,.0f}   "
                  f"${row['fp_cost']:>12,.0f}   "
                  f"${row['expected_credit_loss']:>12,.0f}   "
                  f"{fp_fn_str:>6}")

        print()
        print("=" * 100)
        print("KEY INSIGHTS")
        print("=" * 100)

        # Find best threshold for profit
        best_profit_idx = df['net_profit'].idxmax()
        best_profit_row = df.iloc[best_profit_idx]

        print(f"\n✓ MOST PROFITABLE THRESHOLD: {best_profit_row['threshold']:.0%}")
        print(f"  └─ Net Profit: ${best_profit_row['net_profit']:,.0f}")
        print(f"  └─ Profit Margin: {best_profit_row['profit_margin_pct']:.1f}%")
        print(f"  └─ Approval Rate: {best_profit_row['approval_rate_pct']:.1f}%")
        print(f"  └─ Default Rate: {best_profit_row['default_rate_pct']:.2f}%")

        # Find lowest ECL
        lowest_ecl_idx = df['expected_credit_loss'].idxmin()
        lowest_ecl_row = df.iloc[lowest_ecl_idx]

        print(f"\n✓ LOWEST CREDIT RISK THRESHOLD: {lowest_ecl_row['threshold']:.0%}")
        print(f"  └─ ECL: ${lowest_ecl_row['expected_credit_loss']:,.0f}")
        print(f"  └─ Default Rate: {lowest_ecl_row['default_rate_pct']:.2f}%")
        print(f"  └─ Net Profit: ${lowest_ecl_row['net_profit']:,.0f}")

        # Cost efficiency
        print(f"\n✓ COST STRUCTURE COMPARISON:")
        print(f"  • At 5% threshold:")
        print(f"    - FN Cost: ${df.iloc[0]['fn_cost']:,.0f} (Direct losses from {df.iloc[0]['false_negatives']} bad loans)")
        print(f"    - FP Cost: ${df.iloc[0]['fp_cost']:,.0f} (Opportunity cost from {df.iloc[0]['false_positives']} rejected good customers)")
        print(f"    - FP/FN Ratio: {df.iloc[0]['fp_fn_ratio']:.2f} {'(Too conservative - rejecting too many good customers)' if df.iloc[0]['fp_fn_ratio'] > 1 else '(Good balance)'}")
        print(f"    - Cost Ratio (FP×0.1/FN): {df.iloc[0]['cost_fp_fn_ratio']:.2f} {'(Rejecting 10+ good per bad - TOO CONSERVATIVE)' if df.iloc[0]['cost_fp_fn_ratio'] >= 1 else '(Good balance)'}")
        print(f"  • At {df.iloc[-1]['threshold']:.0%} threshold:")
        print(f"    - FN Cost: ${df.iloc[-1]['fn_cost']:,.0f} (Direct losses from {df.iloc[-1]['false_negatives']} bad loans)")
        print(f"    - FP Cost: ${df.iloc[-1]['fp_cost']:,.0f} (Opportunity cost from {df.iloc[-1]['false_positives']} rejected good customers)")
        print(f"    - FP/FN Ratio: {df.iloc[-1]['fp_fn_ratio']:.2f} {'(Too conservative - rejecting too many good customers)' if df.iloc[-1]['fp_fn_ratio'] > 1 else '(Good balance)'}")
        print(f"    - Cost Ratio (FP×0.1/FN): {df.iloc[-1]['cost_fp_fn_ratio']:.2f} {'(Rejecting 10+ good per bad - TOO CONSERVATIVE)' if df.iloc[-1]['cost_fp_fn_ratio'] >= 1 else '(Good balance)'}")

        print("\n✓ COST RATIO EXPLANATION:")
        print("  Cost Ratio = (FP × 0.1) / FN (based on person counts)")
        print("  • The 0.1 multiplier represents the 10:1 cost ratio ($100 FP / $1000 FN)")
        print("  • Net Opportunity Cost = (Cost Ratio - 1) × $1,000")
        print("    - Ratio = 1.0 → (1.0 - 1.0) × $1,000 = $0 net loss (BREAK EVEN)")
        print("    - Ratio = 1.1 → (1.1 - 1.0) × $1,000 = $100 net loss (slightly over)")
        print("    - Ratio = 2.0 → (2.0 - 1.0) × $1,000 = $1,000 net loss (TOO CONSERVATIVE)")
        print("    - Ratio = 3.0 → (3.0 - 1.0) × $1,000 = $2,000 net loss (VERY BAD)")
        print("  • Ratio < 1.0: Efficient balance (GOOD)")
        print("  • Ratio ≥ 2.0: Too conservative - $1,000+ net loss per bad customer caught (ADJUST THRESHOLD)")
        print(f"\n  At optimal threshold ({best_profit_row['threshold']:.0%}): Cost Ratio = {best_profit_row['cost_fp_fn_ratio']:.2f}")
        net_opp_cost = max(0, (best_profit_row['cost_fp_fn_ratio'] - 1.0) * 1000)
        print(f"    → Net opportunity cost: ${net_opp_cost:,.0f} per FN prevented")
        
        print("\n✓ PROFITABILITY ANALYSIS:")
        print("  Decision Rule: Cost Ratio < 2.0 AND Net Profit > 0 = VIABLE")
        print()
        print("  Cost Ratio Formula: (FP × 0.1) / FN")
        print("  • FP × 0.1 = Effective cost weight of rejected good customers")
        print("  • FN = Number of approved bad customers")
        print()
        print("  Why BOTH conditions matter:")
        print("  • Cost Ratio < 2.0 = Not rejecting too many good customers (< 20 per bad)")
        print("  • Net Profit > 0 = Total revenue exceeds total costs (sustainable)")
        print()
        print("  The Problem:")
        print("  • Low Cost Ratio (< 1.0) just means you're catching bad customers efficiently")
        print("  • BUT if you approve too many defaults, FN cost gets so high that")
        print("    total costs exceed revenue → NEGATIVE PROFIT even with good ratio")
        print()
        
        # Find thresholds that are BOTH profitable AND have good cost ratio
        profitable_thresholds = df[(df['cost_fp_fn_ratio'] < 2.0) & (df['net_profit'] > 0)]
        unprofitable_low_ratio = df[(df['cost_fp_fn_ratio'] < 2.0) & (df['net_profit'] <= 0)]
        too_conservative = df[(df['cost_fp_fn_ratio'] >= 2.0) & (df['cost_fp_fn_ratio'] < float('inf'))]
        
        print("  VIABLE THRESHOLDS (Cost Ratio < 2.0 AND Profitable):")
        print("  " + "-" * 80)
        if len(profitable_thresholds) > 0:
            for _, row in profitable_thresholds.iterrows():
                status = "✓ EXCELLENT" if row['net_profit'] > 2000000 else "✓ GOOD"
                print(f"  {row['threshold']:>5.0%}  | Cost Ratio: {row['cost_fp_fn_ratio']:>6.2f} | Net Profit: ${row['net_profit']:>12,.0f} | {status}")
        else:
            print("  ⚠ No thresholds meet BOTH criteria")
        
        print()
        print("  BAD THRESHOLDS (Cost Ratio < 2.0 BUT Unprofitable - Approving Too Many Defaults):")
        print("  " + "-" * 80)
        if len(unprofitable_low_ratio) > 0:
            for _, row in unprofitable_low_ratio.head(5).iterrows():
                print(f"  {row['threshold']:>5.0%}  | Cost Ratio: {row['cost_fp_fn_ratio']:>6.2f} | Net Profit: ${row['net_profit']:>12,.0f} | ❌ UNPROFITABLE")
        else:
            print("  (None)")
        
        print()
        print("  TOO CONSERVATIVE (Cost Ratio ≥ 2.0 - Rejecting 20+ Good Customers Per Bad):")
        print("  " + "-" * 80)
        if len(too_conservative) > 0:
            for _, row in too_conservative.head(5).iterrows():
                status = "⚠ PROFITABLE" if row['net_profit'] > 0 else "❌ UNPROFITABLE"
                print(f"  {row['threshold']:>5.0%}  | Cost Ratio: {row['cost_fp_fn_ratio']:>6.2f} | Net Profit: ${row['net_profit']:>12,.0f} | {status}")
        
        if len(profitable_thresholds) > 0:
            best = profitable_thresholds.loc[profitable_thresholds['net_profit'].idxmax()]
            print()
            print(f"  ✓ RECOMMENDED THRESHOLD: {best['threshold']:.0%}")
            print(f"  └─ Cost Ratio: {best['cost_fp_fn_ratio']:.2f} (balanced - not over-rejecting)")
            print(f"  └─ Net Profit: ${best['net_profit']:,.0f} (PROFITABLE)")
            print(f"  └─ FP Cost: ${best['fp_cost']:,.0f} vs FN Cost: ${best['fn_cost']:,.0f}")
            print(f"  └─ Approval Rate: {best['approval_rate_pct']:.1f}%")
        else:
            print()
            print(f"  ⚠ NO VIABLE THRESHOLDS - All options either:")
            print(f"     1. Too conservative (high FP cost from rejecting good customers), OR")
            print(f"     2. Too lenient (high FN cost from approving bad customers)")
            print(f"     → Consider adjusting FP/FN cost parameters or improving model")

        print("\n" + "=" * 100)


def main():
    """
    Run threshold stress test on XGBoost model predictions using a representative sample.
    """
    print("Loading dataset and generating predictions...")

    # Load the full synthetic dataset directly
    import pandas as pd
    import glob

    # Find the largest synthetic dataset file
    synthetic_files = glob.glob('data/synthetic/synthetic_credit_data_*.csv')
    if not synthetic_files:
        synthetic_files = ['data/synthetic/synthetic_credit_data.csv']

    # Sort by file size to get the largest
    largest_file = max(synthetic_files, key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0)

    print(f"Loading dataset from: {largest_file}")
    df = pd.read_csv(largest_file)
    total_samples = len(df)
    print(f"✓ Loaded {total_samples:,} samples from synthetic data")

    # Use a representative sample for stress test (100K samples is sufficient for analysis)
    sample_size = min(100000, total_samples)
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"✓ Using {sample_size:,} representative samples for stress test")

    # Apply the same preprocessing as training pipeline
    print("Applying feature preprocessing (one-hot encoding)...")

    # One-hot encode pay_frequency (same as data preparation)
    df_encoded = pd.get_dummies(df_sample, columns=['pay_frequency'], prefix='pay_freq', drop_first=False)

    # Extract target
    target_col = 'Repayment_Status'
    y_full = df_encoded[target_col].values

    # Load XGBoost model to get exact feature names it was trained on
    import joblib
    model = joblib.load('models/xgboost_primary.pkl')

    # Load the feature names from the processed data cache
    processed_data = np.load('data/models_output/processed_data.npz', allow_pickle=True)
    expected_features = processed_data['feature_names']

    print(f"Model expects {len(expected_features)} features: {list(expected_features)}")
    print(f"Current data has {len(df_encoded.columns)} columns")

    # Ensure we have exactly the same features in the same order
    missing_features = set(expected_features) - set(df_encoded.columns)
    extra_features = set(df_encoded.columns) - set(expected_features) - {target_col, 'Max_Safe_Loan_Amount', 'Monthly_Subscription_Fee'}

    if missing_features:
        print(f"⚠ Adding missing features with zeros: {missing_features}")
        for feat in missing_features:
            df_encoded[feat] = 0

    if extra_features:
        print(f"⚠ Removing extra features: {extra_features}")

    # Select only the features the model was trained on, in the correct order
    X_full = df_encoded[expected_features].values

    print(f"✓ Features prepared: {X_full.shape[1]} columns (matches training pipeline exactly)")

    print(f"Generating predictions for {len(X_full):,} samples...")

    # Get predictions on sample dataset
    y_pred_proba = model.predict_proba(X_full)[:, 1]

    print(f"✓ Predictions complete for {len(y_full):,} samples")
    print(f"  - Default rate in data: {y_full.mean():.2%}")
    print()

    # Initialize stress test with full 5M dataset
    stress_test = ThresholdStressTest(
        y_true=y_full,
        y_pred_proba=y_pred_proba,
        avg_loan_amount=700,
        subscription_ltv=106.92,  # 6 months × $17.82
        cost_per_default=1000
    )

    # Run analysis with 5% increments from 5% to 100%
    print(f"Running stress test for thresholds: 5% to 100% in 5% increments")
    print(f"Analyzing {len(y_full):,} representative samples (statistically significant sample)")
    print()

    results_df = stress_test.run_stress_test()  # Uses default range

    # Generate outputs
    print("Generating comparison tables and visualizations...")
    display_table = stress_test.generate_comparison_table(results_df)
    print("✓ Tables generated and saved")

    stress_test.plot_stress_test_results(results_df)
    print("✓ Line graphs generated and saved")

    stress_test.plot_heatmap_comparison(results_df)
    print("✓ Heatmap visualizations generated and saved")

    stress_test.print_summary_report(results_df)

    print("\n" + "=" * 100)
    print("OUTPUT FILES SAVED")
    print("=" * 100)
    print(f"📊 Comparison Table (formatted): data/stress_test/threshold_comparison_table.csv")
    print(f"📊 Comparison Table (raw data):  data/stress_test/threshold_comparison_raw.csv")
    print(f"📈 Dual-Axis Graph:              data/stress_test/threshold_stress_test_dual_axis.png")
    print(f"📈 Comprehensive Graphs:         data/stress_test/threshold_stress_test_comprehensive.png")
    print(f"🔥 Heatmap (All Metrics):        data/stress_test/threshold_heatmap_comparison.png")
    print(f"🔥 Heatmap (Profit vs ECL):      data/stress_test/threshold_profit_vs_ecl_heatmap.png")
    print("=" * 100)
    print()

    return results_df, display_table


if __name__ == "__main__":
    results_df, display_table = main()