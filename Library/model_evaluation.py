
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrixAnalyzer:
    """
    Comprehensive confusion matrix analysis for Reverse Savings credit model.
    
    Business Context (Standard Statistical Convention):
    - Positive (1): Customer DEFAULTS (bad customer)
    - Negative (0): Customer REPAYS (good customer)
    
    Key Metrics:
    - Accuracy: Overall correctness
    - Precision: Of predicted defaults, how many actually defaulted?
    - Recall: Of actual defaults, how many did we predict?
    - False Positive Rate: Of customers who repaid, how many did we incorrectly predict as default?
    """
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize with predictions.
        
        Args:
            y_true: Actual labels (0=Repays, 1=Default)
            y_pred: Predicted labels (0=No Default, 1=Default)
            y_pred_proba: Predicted probabilities (optional, for AUC-ROC)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        # Calculate confusion matrix
        # Standard layout: rows=actual, cols=predicted
        # labels=[1, 0] creates matrix:
        #         pred=1  pred=0
        # actual=1  TP      FN
        # actual=0  FP      TN
        self.cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        
        # Extract components correctly from the matrix
        # cm[0,0]=TP, cm[0,1]=FN, cm[1,0]=FP, cm[1,1]=TN
        self.tp = self.cm[0, 0]  # actual=1, pred=1
        self.fn = self.cm[0, 1]  # actual=1, pred=0
        self.fp = self.cm[1, 0]  # actual=0, pred=1
        self.tn = self.cm[1, 1]  # actual=0, pred=0
        
    def calculate_all_metrics(self):
        """
        Calculate all business-critical metrics.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # 1. ACCURACY - Overall correctness
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        
        # 2. PRECISION - Of predicted defaults, what % actually defaulted?
        # Critical for business: High precision = we're predicting defaults accurately
        metrics['precision'] = precision_score(self.y_true, self.y_pred, pos_label=1)
        
        # 3. RECALL - Of actual defaults, what % did we correctly predict?
        # Critical for risk: High recall = we're catching most defaults
        # Explicitly verify the calculation matches manual formula
        metrics['recall'] = recall_score(self.y_true, self.y_pred, pos_label=1, zero_division=0)
        
        # Manual verification (should match sklearn)
        manual_recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        if abs(metrics['recall'] - manual_recall) > 0.001:
            print(f"⚠ WARNING: Recall mismatch - sklearn: {metrics['recall']:.4f}, manual: {manual_recall:.4f}")
            metrics['recall'] = manual_recall  # Use manual calculation if mismatch
        
        # 4. FALSE POSITIVE RATE - Of rejected customers, what % would have repaid?
        # Critical for opportunity cost: Low FPR = not rejecting good customers
        metrics['false_positive_rate'] = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
        
        # 5. F1-SCORE - Harmonic mean of precision and recall
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred)
        
        # 6. AUC-ROC - Model's discriminatory power
        if self.y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(self.y_true, self.y_pred_proba)
        
        # 7. TRUE NEGATIVE RATE (Specificity) - Of customers who would default, what % did we reject?
        metrics['true_negative_rate'] = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
        
        return metrics
    
    def print_business_interpretation(self):
        """
        Print confusion matrix with business interpretation.
        """
        metrics = self.calculate_all_metrics()
        
        print("=" * 70)
        print("REVERSE SAVINGS - CONFUSION MATRIX ANALYSIS")
        print("=" * 70)
        print()
        
        print("CONFUSION MATRIX:")
        print("-" * 70)
        print(f"                           Actual: Default (1)  |  Actual: Repays (0)")
        print(f"Predicted: Default (1)          TP = {self.tp:,}         |  FP = {self.fp:,}")
        print(f"Predicted: No Default (0)       FN = {self.fn:,}         |  TN = {self.tn:,}")
        print()
        
        print("BUSINESS INTERPRETATION:")
        print("-" * 70)
        print(f"• TRUE POSITIVES (TP):  {self.tp:,} - Correctly predicted defaults ✓")
        print(f"• TRUE NEGATIVES (TN):  {self.tn:,} - Correctly predicted repayments ✓")
        print(f"• FALSE POSITIVES (FP): {self.fp:,} - Incorrectly predicted default (good customer rejected)")
        print(f"• FALSE NEGATIVES (FN): {self.fn:,} - Incorrectly predicted repayment (bad customer approved - Financial Loss)")
        print()
        
        print("KEY PERFORMANCE METRICS:")
        print("-" * 70)
        print(f"1. ACCURACY:             {metrics['accuracy']:.2%}")
        print(f"   → Overall correctness of predictions")
        print()
        
        print(f"2. PRECISION:            {metrics['precision']:.2%}")
        print(f"   → Of predicted defaults, {metrics['precision']:.2%} actually defaulted")
        print(f"   → Formula: TP/(TP+FP) = {self.tp}/({self.tp}+{self.fp}) = {self.tp}/{self.tp+self.fp}")
        print(f"   → Business Impact: Accuracy of default predictions")
        print()
        
        print(f"3. RECALL:               {metrics['recall']:.2%}")
        print(f"   → Of actual defaults, we correctly predicted {metrics['recall']:.2%}")
        print(f"   → Formula: TP/(TP+FN) = {self.tp}/({self.tp}+{self.fn}) = {self.tp}/{self.tp+self.fn}")
        print(f"   → Business Impact: Default detection rate")
        print()
        
        print(f"4. FALSE POSITIVE RATE:  {metrics['false_positive_rate']:.2%}")
        print(f"   → Of customers who repaid, {metrics['false_positive_rate']:.2%} were incorrectly predicted as default")
        print(f"   → Business Impact: Opportunity cost (good customers rejected)")
        print()
        
        print(f"5. F1-SCORE:             {metrics['f1_score']:.2%}")
        print(f"   → Harmonic mean of precision and recall")
        print(f"   → Business Impact: Overall model quality (balances risk vs growth)")
        print(f"   → Target: >80% for production deployment")
        print()
        
        if 'auc_roc' in metrics:
            print(f"6. AUC-ROC:              {metrics['auc_roc']:.4f}")
            print(f"   → Model's ability to distinguish repayers from defaulters")
            print()
        
        print("FINANCIAL IMPACT:")
        print("-" * 70)
        total_approved = self.tp + self.fn
        total_defaults = self.fn
        default_rate = (self.fn / total_approved * 100) if total_approved > 0 else 0
        
        print(f"• Total Approved:     {total_approved:,}")
        print(f"• Total Defaults:     {total_defaults:,}")
        print(f"• Default Rate:       {default_rate:.2f}%")
        print(f"• Success Rate:       {100-default_rate:.2f}%")
        print()
        
        # Federal comparison
        print("FEDERAL BENCHMARK COMPARISON:")
        print("-" * 70)
        print(f"• Federal student loans default rate: ~10-15%")
        print(f"• Reverse Savings default rate:        {default_rate:.2f}%")
        
        if default_rate < 15:
            print(f"  ✓ BETTER than federal government! ({15-default_rate:.1f}% improvement)")
        else:
            print(f"  ⚠ Need improvement to match federal standards")
        
        print()
        print("=" * 70)
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Visualize confusion matrix with business labels.
        """
        plt.figure(figsize=(10, 8))
        
        # Create labels (TP/FP first row, FN/TN second row)
        labels = np.array([
            [f'TP\n{self.tp:,}\n(Correct Default)', f'FP\n{self.fp:,}\n(False Alarm - Opportunity Lost)'],
            [f'FN\n{self.fn:,}\n(Missed Default - Financial Loss)', f'TN\n{self.tn:,}\n(Correct Repayment)']
        ])
        
        # Plot heatmap
        sns.heatmap(self.cm, annot=labels, fmt='', cmap='Blues', 
                    xticklabels=['Default (1)', 'Repays (0)'],
                    yticklabels=['Predicted: Default (1)', 'Predicted: No Default (0)'],
                    cbar_kws={'label': 'Count'})
        
        plt.title('Reverse Savings - Confusion Matrix\n(15% Per-Paycheck Subscription Model)', 
                  fontsize=14, fontweight='bold')
        plt.ylabel('Predicted Status', fontsize=12)
        plt.xlabel('Actual Status', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Simulate predictions (replace with actual model predictions)
    np.random.seed(42)
    n_samples = 5000
    
    # Simulate ground truth (32% default rate, 68% repayment rate from synthetic data)
    # 0 = Repays, 1 = Default
    y_true = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])
    
    # Simulate predictions (model with 85% accuracy)
    y_pred = y_true.copy()
    flip_indices = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
    y_pred[flip_indices] = 1 - y_pred[flip_indices]
    
    # Simulate probabilities (probability of default)
    y_pred_proba = np.where(y_true == 1,  # If actually defaults
                            np.random.beta(8, 2, n_samples),  # High probability
                            np.random.beta(2, 8, n_samples))  # Low probability
    
    # Analyze
    analyzer = ConfusionMatrixAnalyzer(y_true, y_pred, y_pred_proba)
    metrics = analyzer.print_business_interpretation()
    analyzer.plot_confusion_matrix(save_path='data/confusion_matrix.png')
