"""
Cost-Based Threshold Optimization
Determines optimal decision threshold based on business costs of false positives vs false negatives

BUSINESS COST DEFINITIONS:
==========================

FALSE NEGATIVE (FN) = Approving a Bad Customer Who Defaults
- Direct financial loss: Average loan amount ($700) + recovery costs
- Total cost: $1,000 per bad loan
- This is a REAL LOSS - money that cannot be recovered

FALSE POSITIVE (FP) = Rejecting a Good Customer Who Would Have Repaid
- Opportunity cost: Lost subscription revenue over customer lifetime
- NOT a direct loss, but foregone profit
- Calculation (from Library/revenue_projections.py):
    * Base subscription: $16.12/month
    * Premium add-ons (85% adoption): ~$1.70/month
    * Average revenue: ~$17.82/month
    * Over 5-6 months: $17.82 × 6 = ~$107
    * Conservative estimate: $100 per rejected good customer

COST RATIO INTERPRETATION:
=========================
- Current ratio: 10:1 (cost_fn / cost_fp = $1,000 / $100 = 10)
- Meaning: You can afford to reject 10 good customers (losing $1,000 in subscription revenue)
  before it equals the cost of 1 bad loan default ($1,000 direct loss)
- This is intentionally CONSERVATIVE because:
    1. Direct losses hurt more than opportunity costs
    2. Can acquire new customers, but can't recover defaulted loans
    3. Bad loans damage reputation and investor confidence

ADJUSTING PARAMETERS:
====================
- Increase cost_fp if customer LTV is higher (longer retention, more upsells)
- Decrease cost_fp if acquisition costs are high (harder to replace rejected customers)
- Increase cost_fn if loan amounts grow or recovery costs increase
- These parameters should be calibrated based on actual business data
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os


class CostBasedThresholdOptimizer:
    """
    Optimize decision threshold based on asymmetric business costs.

    Business Context:
    - False Negative (FN) = Direct financial loss from defaulted loan (~$1,000)
    - False Positive (FP) = Opportunity cost from lost subscription revenue (~$100)
    - Ratio = 10:1, meaning we're conservative about approvals to protect capital
    """

    def __init__(self, cost_fn=1000, cost_fp=100):
        """
        Initialize optimizer with business costs.

        Args:
            cost_fn: Cost of False Negative (approving bad customer who defaults)
                    Default $1,000 = avg loan ($700) + recovery costs ($300)
            cost_fp: Cost of False Positive (rejecting good customer)
                    Default $100 = ~5-6 months lost subscription revenue

        Note: 
            The 10:1 ratio (cost_fn/cost_fp) means you can reject 10 good customers
            before it costs as much as 1 bad loan. This is intentionally conservative
            because direct losses are harder to recover than opportunity costs.
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def calculate_total_cost(self, y_true, y_pred):
        """
        Calculate the total business cost for a given set of predictions.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Cost = (FN * Cost_FN) + (FP * Cost_FP)
        total_cost = (fn * self.cost_fn) + (fp * self.cost_fp)
        return total_cost, fn, fp

    def find_optimal_threshold(self, y_true, y_pred_proba):
        """
        Find the optimal decision threshold that minimizes total business cost.

        Args:
            y_true: True labels (0 or 1)
            y_pred_proba: Predicted probabilities for the positive class (1)

        Returns:
            optimal_threshold: The threshold that minimizes total cost
            min_cost: The minimum total cost achievable
            optimal_fn: Number of False Negatives at the optimal threshold
            optimal_fp: Number of False Positives at the optimal threshold
            thresholds: Array of thresholds tested
            total_costs: Array of total costs for each threshold
            fn_counts: Array of False Negative counts for each threshold
            fp_counts: Array of False Positive counts for each threshold
        """
        thresholds = np.linspace(0, 1, 101)  # Test thresholds from 0 to 1
        min_cost = float('inf')
        optimal_threshold = -1
        optimal_fn = -1
        optimal_fp = -1

        total_costs = []
        fn_counts = []
        fp_counts = []

        for threshold in thresholds:
            # Convert probabilities to binary predictions based on threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix and costs
            cost, fn, fp = self.calculate_total_cost(y_true, y_pred)
            
            total_costs.append(cost)
            fn_counts.append(fn)
            fp_counts.append(fp)

            # Update optimal threshold if current cost is lower
            if cost < min_cost:
                min_cost = cost
                optimal_threshold = threshold
                optimal_fn = fn
                optimal_fp = fp

        return optimal_threshold, min_cost, optimal_fn, optimal_fp, thresholds, total_costs, fn_counts, fp_counts

    def plot_cost_curve(self, thresholds, total_costs, fn_counts, fp_counts, optimal_threshold, min_cost, optimal_fn, optimal_fp):
        """
        Plot the cost curve and highlight the optimal threshold.
        """
        plt.figure(figsize=(12, 8))

        # Plot total cost curve
        plt.subplot(2, 1, 1)
        plt.plot(thresholds, total_costs, label='Total Cost')
        plt.scatter(optimal_threshold, min_cost, color='red', zorder=5, label=f'Optimal Threshold ({optimal_threshold:.3f})')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Total Business Cost ($)')
        plt.title('Cost vs. Decision Threshold')
        plt.legend()
        plt.grid(True)

        # Plot FN and FP counts
        plt.subplot(2, 1, 2)
        plt.plot(thresholds, fn_counts, label='False Negatives (FN)', color='orange')
        plt.plot(thresholds, fp_counts, label='False Positives (FP)', color='green')
        plt.scatter(optimal_threshold, optimal_fn, color='red', zorder=5, marker='o', label=f'Optimal FN ({optimal_fn})')
        plt.scatter(optimal_threshold, optimal_fp, color='red', zorder=5, marker='x', label=f'Optimal FP ({optimal_fp})')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Number of Occurrences')
        plt.title('False Negatives vs. False Positives')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def display_results(self, optimal_threshold, min_cost, optimal_fn, optimal_fp, thresholds, total_costs, fn_counts, fp_counts):
        """
        Display a summary of the cost-based optimization results.
        """
        print("=" * 70)
        print("COST-BASED THRESHOLD OPTIMIZATION")
        print("=" * 70)
        print("Business Cost Structure:")
        print("-" * 70)
        print(f"False Negative (FN) Cost: ${self.cost_fn:,.2f}")
        print(f"  └─ Direct financial loss from defaulted loan")
        print(f"  └─ Includes: Loan principal + recovery costs")
        print()
        print(f"False Positive (FP) Cost: ${self.cost_fp:,.2f}")
        print(f"  └─ Opportunity cost from lost subscription revenue")
        print(f"  └─ Represents: ~{self.cost_fp/17.82:.1f} months of avg subscription")
        print()
        print(f"Cost Ratio (FN:FP): {self.cost_fn/self.cost_fp:.1f}:1")
        print(f"  └─ Can reject {int(self.cost_fn/self.cost_fp)} good customers before it equals 1 bad loan")
        print()

        print(f"\nOPTIMAL THRESHOLD: {optimal_threshold:.3f}")
        print("-" * 70)
        print("Expected Outcomes at Optimal Threshold:")
        print(f"  False Negatives (FN): {int(optimal_fn)} bad loans approved")
        print(f"    └─ Direct Financial Loss: ${optimal_fn * self.cost_fn:,.2f}")
        print()
        print(f"  False Positives (FP): {int(optimal_fp)} good customers rejected")
        print(f"    └─ Opportunity Cost (Lost Revenue): ${optimal_fp * self.cost_fp:,.2f}")
        print()
        print(f"Total Expected Cost: ${min_cost:,.2f}")
        print(f"  └─ {(optimal_fn * self.cost_fn / min_cost * 100):.1f}% from direct losses (bad loans)")
        print(f"  └─ {(optimal_fp * self.cost_fp / min_cost * 100):.1f}% from opportunity costs (rejected customers)")
        print()

        # Compare with default threshold
        default_threshold = 0.5
        default_idx = np.argmin(np.abs(thresholds - default_threshold))
        default_cost = total_costs[default_idx]
        default_fn = fn_counts[default_idx]
        default_fp = fp_counts[default_idx]

        print(f"COMPARISON WITH DEFAULT THRESHOLD (0.5):")
        print("-" * 70)
        print(f"Default Total Cost: ${default_cost:,.2f}")
        print(f"  └─ {int(default_fn)} bad loans (${default_fn * self.cost_fn:,.2f} direct loss)")
        print(f"  └─ {int(default_fp)} rejected customers (${default_fp * self.cost_fp:,.2f} opportunity cost)")
        print()
        print(f"Optimized Total Cost: ${min_cost:,.2f}")
        print(f"Cost Savings: ${default_cost - min_cost:,.2f} ({(default_cost - min_cost)/default_cost*100:.1f}%)")
        print()
        print("Business Interpretation:")
        if optimal_fn < default_fn:
            print(f"  ✓ Fewer bad loans approved ({int(default_fn - optimal_fn)} reduction)")
            print(f"    → Protecting ${(default_fn - optimal_fn) * self.cost_fn:,.2f} in capital")
        if optimal_fp > default_fp:
            print(f"  ⚠ More good customers rejected ({int(optimal_fp - default_fp)} increase)")
            print(f"    → But opportunity cost (${(optimal_fp - default_fp) * self.cost_fp:,.2f}) < capital saved")
        print()

# Example Usage (assuming you have y_true and y_pred_proba from a model)
if __name__ == '__main__':
    # Dummy data for demonstration
    # Let's simulate a scenario where:
    # - 1000 samples
    # - 100 actual defaults (FNs are more costly)
    # - 900 actual good customers (FPs are less costly)
    n_samples = 1000
    cost_fn_example = 1000
    cost_fp_example = 100

    np.random.seed(42)

    # Simulate true labels: more good customers than bad
    y_true_dummy = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    # Simulate predicted probabilities
    # For good customers (y_true=0), assign lower probabilities
    # For bad customers (y_true=1), assign higher probabilities
    y_pred_proba_dummy = np.zeros(n_samples)
    
    # Assign probabilities for good customers (y_true=0)
    good_customer_indices = np.where(y_true_dummy == 0)[0]
    y_pred_proba_dummy[good_customer_indices] = np.random.uniform(0, 0.7, size=len(good_customer_indices)) # Mostly low probs, some overlap

    # Assign probabilities for bad customers (y_true=1)
    bad_customer_indices = np.where(y_true_dummy == 1)[0]
    y_pred_proba_dummy[bad_customer_indices] = np.random.uniform(0.3, 1.0, size=len(bad_customer_indices)) # Mostly high probs, some overlap

    # Introduce some noise/overlap
    y_pred_proba_dummy = np.clip(y_pred_proba_dummy, 0, 1)


    # Initialize the optimizer
    optimizer = CostBasedThresholdOptimizer(cost_fn=cost_fn_example, cost_fp=cost_fp_example)

    # Find the optimal threshold
    optimal_threshold, min_cost, optimal_fn, optimal_fp, thresholds, total_costs, fn_counts, fp_counts = optimizer.find_optimal_threshold(y_true_dummy, y_pred_proba_dummy)

    # Display results and plot the cost curve
    optimizer.display_results(optimal_threshold, min_cost, optimal_fn, optimal_fp, thresholds, total_costs, fn_counts, fp_counts)
    optimizer.plot_cost_curve(thresholds, total_costs, fn_counts, fp_counts, optimal_threshold, min_cost, optimal_fn, optimal_fp)

    print("\n--- Custom Costs Example ---")
    # Example with different costs
    custom_cost_fn = 1500  # Higher cost for bad loans
    custom_cost_fp = 200   # Higher opportunity cost for rejecting good customers

    optimizer_custom = CostBasedThresholdOptimizer(cost_fn=custom_cost_fn, cost_fp=custom_cost_fp)
    optimal_threshold_custom, min_cost_custom, optimal_fn_custom, optimal_fp_custom, thresholds_custom, total_costs_custom, fn_counts_custom, fp_counts_custom = optimizer_custom.find_optimal_threshold(y_true_dummy, y_pred_proba_dummy)

    optimizer_custom.display_results(optimal_threshold_custom, min_cost_custom, optimal_fn_custom, optimal_fp_custom, thresholds_custom, total_costs_custom, fn_counts_custom, fp_counts_custom)
    optimizer_custom.plot_cost_curve(thresholds_custom, total_costs_custom, fn_counts_custom, fp_counts_custom, optimal_threshold_custom, min_cost_custom, optimal_fn_custom, optimal_fp_custom)