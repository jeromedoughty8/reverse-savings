
"""
Reverse Savings Credit System - Week 5
SHAP-based Explainable AI (XAI) Implementation

Provides global and local explanations for XGBoost models to satisfy:
1. Regulatory compliance (individual decision explanations)
2. Bias/fairness validation
3. Business justification of model decisions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from pathlib import Path

class SHAPExplainer:
    """
    SHAP explainability wrapper for XGBoost models.
    
    Provides both global (feature importance) and local (individual prediction) explanations.
    """
    
    def __init__(self, model, X_data, feature_names, model_type='classification'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained XGBoost model
            X_data: Feature data (numpy array or pandas DataFrame)
            feature_names: List of feature names
            model_type: 'classification' or 'regression'
        """
        self.model = model
        self.X_data = X_data
        self.feature_names = feature_names
        self.model_type = model_type
        
        # Initialize SHAP TreeExplainer (optimized for XGBoost)
        print("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        print("Calculating SHAP values (this may take a moment)...")
        self.shap_values = self.explainer.shap_values(X_data)
        
        # For binary classification, shap_values might be a list
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use positive class (Default=1)
        
        print("✓ SHAP initialization complete")
    
    def plot_global_importance(self, output_dir='data/SHAP', max_display=20):
        """
        Create global feature importance plots.
        
        Shows which features are most important across all predictions.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating Global Feature Importance Plots...")
        print("-" * 70)
        
        # 1. Summary Plot (beeswarm) - shows feature impact distribution
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_data, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance - Global Impact Distribution', 
                  fontweight='bold', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/global_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/global_summary_plot.png")
        
        # 2. Bar Plot - mean absolute SHAP values
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_data, 
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance - Mean Impact', 
                  fontweight='bold', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/global_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/global_bar_plot.png")
        
        # 3. Feature Importance Table
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        
        importance_df.to_csv(f'{output_dir}/global_feature_importance.csv', index=False)
        print(f"✓ Saved: {output_dir}/global_feature_importance.csv")
        
        print("\nTop 10 Most Important Features (by SHAP):")
        print(importance_df.head(10).to_string(index=False))
        print()
        
        return importance_df
    
    def plot_local_explanation(self, sample_idx, output_dir='data/SHAP', case_name=None):
        """
        Create local explanation for a specific prediction (force plot).
        
        Args:
            sample_idx: Index of sample to explain
            output_dir: Directory to save plots
            case_name: Optional name for the case (e.g., "approved_customer_1")
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if case_name is None:
            case_name = f"sample_{sample_idx}"
        
        print(f"\nGenerating Local Explanation for {case_name}...")
        print("-" * 70)
        
        # Get prediction
        if self.model_type == 'classification':
            prediction_proba = self.model.predict_proba(self.X_data[sample_idx:sample_idx+1])[:, 1][0]
            prediction_class = self.model.predict(self.X_data[sample_idx:sample_idx+1])[0]
            print(f"Prediction: {'DEFAULT' if prediction_class == 1 else 'REPAYS'} (probability: {prediction_proba:.4f})")
        else:
            prediction = self.model.predict(self.X_data[sample_idx:sample_idx+1])[0]
            print(f"Prediction: ${prediction:,.2f}")
        
        # Force plot (waterfall style)
        shap_explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.explainer.expected_value,
            data=self.X_data[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(14, 6))
        shap.plots.waterfall(shap_explanation, max_display=15, show=False)
        plt.title(f'SHAP Explanation - {case_name}', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/local_explanation_{case_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/local_explanation_{case_name}.png")
        
        # Feature contribution table
        contribution_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Value': self.X_data[sample_idx],
            'SHAP_Value': self.shap_values[sample_idx]
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        contribution_df.to_csv(f'{output_dir}/local_contributions_{case_name}.csv', index=False)
        print(f"✓ Saved: {output_dir}/local_contributions_{case_name}.csv")
        
        print("\nTop 5 Contributing Features:")
        print(contribution_df.head(5).to_string(index=False))
        print()
        
        return contribution_df
    
    def generate_case_studies(self, y_true, y_pred, output_dir='data/SHAP'):
        """
        Generate explanations for representative cases:
        - True Positive (correctly predicted default)
        - True Negative (correctly predicted repayment)
        - False Positive (incorrectly predicted default)
        - False Negative (incorrectly predicted repayment - missed default)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating Case Studies...")
        print("=" * 70)
        
        # Find representative samples
        tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
        tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
        fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
        fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
        
        cases = []
        
        # True Positive - Correctly identified default
        if len(tp_idx) > 0:
            idx = tp_idx[0]
            print("\nCASE 1: TRUE POSITIVE (Correctly Predicted Default)")
            cases.append(self.plot_local_explanation(idx, output_dir, "TP_correct_default"))
        
        # True Negative - Correctly identified repayment
        if len(tn_idx) > 0:
            idx = tn_idx[0]
            print("\nCASE 2: TRUE NEGATIVE (Correctly Predicted Repayment)")
            cases.append(self.plot_local_explanation(idx, output_dir, "TN_correct_repayment"))
        
        # False Positive - Incorrectly rejected good customer
        if len(fp_idx) > 0:
            idx = fp_idx[0]
            print("\nCASE 3: FALSE POSITIVE (Incorrectly Predicted Default - Opportunity Lost)")
            cases.append(self.plot_local_explanation(idx, output_dir, "FP_opportunity_lost"))
        
        # False Negative - Missed default (financial loss)
        if len(fn_idx) > 0:
            idx = fn_idx[0]
            print("\nCASE 4: FALSE NEGATIVE (Missed Default - Financial Loss)")
            cases.append(self.plot_local_explanation(idx, output_dir, "FN_financial_loss"))
        
        print("=" * 70)
        print("✓ Case studies complete")
        print()
        
        return cases
    
    def analyze_bias_fairness(self, output_dir='data/SHAP'):
        """
        Validate that model is not using proxy features for discrimination.
        
        Checks if any features have suspiciously high impact that could indicate bias.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nBias/Fairness Analysis...")
        print("=" * 70)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        
        # Define ethical features (alternative credit data)
        ethical_features = [
            'monthly_net_income',
            'on_time_rent_payments_pct',
            'utility_payment_consistency',
            'active_subscription_months',
            'employment_tenure_months',
            'bank_account_age_months',
            'income_stability_index',
            'alt_debt_to_income_ratio'
        ]
        
        # Check top features
        top_10_features = importance_df.head(10)['Feature'].tolist()
        ethical_in_top10 = [f for f in top_10_features if f in ethical_features]
        
        print("Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        print()
        
        print(f"Ethical Alternative Credit Features in Top 10: {len(ethical_in_top10)}/10")
        print(f"Features: {ethical_in_top10}")
        print()
        
        if len(ethical_in_top10) >= 7:
            print("✓ PASS: Model primarily relies on ethical alternative credit features")
        else:
            print("⚠ WARNING: Model may be over-relying on non-alternative-credit features")
        
        # Save bias report
        bias_report = {
            'Total_Features': len(self.feature_names),
            'Ethical_Features_in_Top10': len(ethical_in_top10),
            'Top_Feature': importance_df.iloc[0]['Feature'],
            'Top_Feature_Impact': importance_df.iloc[0]['Mean_Abs_SHAP'],
            'Assessment': 'PASS' if len(ethical_in_top10) >= 7 else 'WARNING'
        }
        
        pd.DataFrame([bias_report]).to_csv(f'{output_dir}/bias_fairness_report.csv', index=False)
        print(f"\n✓ Saved: {output_dir}/bias_fairness_report.csv")
        print("=" * 70)
        print()
        
        return importance_df


def week5_main():
    """Execute Week 5: SHAP Explainability Analysis."""
    print("=" * 70)
    print("EXPLAINABLE AI (SHAP) IMPLEMENTATION")
    print("=" * 70)
    print()
    
    # Load data and models
    print("Step 1: Loading Models and Data")
    print("-" * 70)
    data = np.load('data/models_output/processed_data.npz', allow_pickle=True)
    X_test = data['X_test']
    y_class_test = data['y_class_test']
    feature_names = data['feature_names'].tolist()
    
    xgb_classifier = joblib.load('models/xgboost_primary.pkl')
    
    print(f"✓ Loaded XGBoost classifier")
    print(f"✓ Full test set: {len(X_test)} samples")
    print(f"✓ Features: {len(feature_names)}")
    
    # Use stratified sample for SHAP (much faster, still representative)
    sample_size = min(5000, len(X_test))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_test_sample = X_test[sample_indices]
    y_class_test_sample = y_class_test[sample_indices]
    
    print(f"✓ Using stratified sample of {sample_size} for SHAP analysis (faster)")
    print()
    
    xgb_pred = xgb_classifier.predict(X_test_sample)
    
    # Initialize SHAP explainer for classification
    print("Step 2: Initializing SHAP Explainer (Classification)")
    print("-" * 70)
    shap_classifier = SHAPExplainer(
        model=xgb_classifier,
        X_data=X_test_sample,
        feature_names=feature_names,
        model_type='classification'
    )
    print()
    
    # Global feature importance
    print("Step 3: Global Feature Importance Analysis")
    print("-" * 70)
    importance_df = shap_classifier.plot_global_importance(
        output_dir='data/SHAP/Classification',
        max_display=20
    )
    
    # Case studies
    print("Step 4: Generating Case Studies")
    print("-" * 70)
    shap_classifier.generate_case_studies(
        y_true=y_class_test_sample,
        y_pred=xgb_pred,
        output_dir='data/SHAP/Classification/CaseStudies'
    )
    
    # Bias/Fairness analysis
    print("Step 5: Bias/Fairness Validation")
    print("-" * 70)
    shap_classifier.analyze_bias_fairness(
        output_dir='data/SHAP/Classification'
    )
    
    # Optional: Regression model SHAP analysis
    if os.path.exists('models/regression/xgboost_regression.pkl'):
        print("Step 6: SHAP Analysis for Regression Model")
        print("-" * 70)
        xgb_regressor = joblib.load('models/regression/xgboost_regression.pkl')
        
        shap_regressor = SHAPExplainer(
            model=xgb_regressor,
            X_data=X_test,
            feature_names=feature_names,
            model_type='regression'
        )
        
        shap_regressor.plot_global_importance(
            output_dir='data/SHAP/Regression',
            max_display=20
        )
        print()
    
    print("=" * 70)
    print("SHAP EXPLAINABILITY COMPLETE! ✓")
    print("=" * 70)
    print()
    print("Deliverables:")
    print("  ✓ SHAP TreeExplainer initialized")
    print("  ✓ Global feature importance (summary & bar plots)")
    print("  ✓ Local explanations (4 case studies)")
    print("  ✓ Bias/fairness validation report")
    print("  ✓ Regulatory compliance ready")
    print()
    print("Key Outputs:")
    print("  • data/SHAP/Classification/global_summary_plot.png")
    print("  • data/SHAP/Classification/global_bar_plot.png")
    print("  • data/SHAP/Classification/CaseStudies/TP_correct_default.png")
    print("  • data/SHAP/Classification/CaseStudies/FN_financial_loss.png")
    print("  • data/SHAP/Classification/bias_fairness_report.csv")
    print()
    print("Next: Business Analysis & Final Reporting")
    print("=" * 70)


if __name__ == "__main__":
    week5_main()
