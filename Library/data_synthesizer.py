
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class AlternativeCreditSynthesizer:
    """
    Synthesizes alternative credit features (ACF) for the Reverse Savings model.
    These features would normally come from APIs like Plaid in production.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize synthesizer with random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
    
    def synthesize_discipline_features(self, n_samples: int) -> pd.DataFrame:
        """
        Generate discipline-related features that indicate payment reliability.
        
        Returns:
            DataFrame with discipline features
        """
        # On-time rent payments (high = good discipline)
        on_time_rent_payments_pct = np.random.beta(8, 2, n_samples) * 100
        
        # Active subscription months (longer = better discipline)
        active_subscription_months = np.random.gamma(3, 4, n_samples).clip(0, 60)
        
        # Utility payment consistency (0-100 scale)
        utility_payment_consistency = np.random.beta(7, 2, n_samples) * 100
        
        return pd.DataFrame({
            'on_time_rent_payments_pct': on_time_rent_payments_pct,
            'active_subscription_months': active_subscription_months,
            'utility_payment_consistency': utility_payment_consistency
        })
    
    def synthesize_capacity_features(self, n_samples: int) -> pd.DataFrame:
        """
        Generate capacity-related features that indicate ability to repay.
        
        Returns:
            DataFrame with capacity features
        """
        # Monthly net income (lognormal distribution, realistic income range)
        # Centered around $3,500/month with range $1,000-$8,000
        monthly_net_income = np.random.lognormal(8.0, 0.5, n_samples).clip(1000, 8000)
        
        # Pay frequency: 'WEEKLY' (52/year), 'BIWEEKLY' (26/year), 'SEMIMONTHLY' (24/year)
        pay_frequencies = np.random.choice(['BIWEEKLY', 'SEMIMONTHLY', 'WEEKLY'], 
                                          n_samples, 
                                          p=[0.60, 0.30, 0.10])  # Most common is biweekly
        
        # Calculate paycheck amount based on frequency
        paychecks_per_year = np.where(pay_frequencies == 'WEEKLY', 52,
                                     np.where(pay_frequencies == 'BIWEEKLY', 26, 24))
        paycheck_amount = (monthly_net_income * 12) / paychecks_per_year
        
        # Monthly rent/housing cost
        monthly_rent = monthly_net_income * np.random.uniform(0.2, 0.4, n_samples)
        
        # Monthly recurring bills (subscriptions, utilities, etc.)
        monthly_recurring_bills = np.random.uniform(200, 800, n_samples)
        
        # Calculate alternative debt-to-income ratio
        alt_debt_to_income_ratio = (monthly_rent + monthly_recurring_bills) / monthly_net_income
        
        return pd.DataFrame({
            'monthly_net_income': monthly_net_income,
            'pay_frequency': pay_frequencies,
            'paycheck_amount': paycheck_amount,
            'paychecks_per_year': paychecks_per_year,
            'monthly_rent': monthly_rent,
            'monthly_recurring_bills': monthly_recurring_bills,
            'alt_debt_to_income_ratio': alt_debt_to_income_ratio
        })
    
    def synthesize_stability_features(self, n_samples: int, 
                                     monthly_income: np.ndarray) -> pd.DataFrame:
        """
        Generate stability-related features.
        
        Args:
            monthly_income: Array of monthly net income values
            
        Returns:
            DataFrame with stability features
        """
        # Income stability index (inverse of coefficient of variation)
        # High value = stable income, low variance
        income_variance = monthly_income * np.random.uniform(0.05, 0.3, n_samples)
        income_stability_index = 1 / (income_variance / monthly_income).clip(0.01, 10)
        
        # Employment tenure in months
        employment_tenure_months = np.random.gamma(2, 12, n_samples).clip(0, 120)
        
        # Bank account age in months
        bank_account_age_months = np.random.gamma(3, 15, n_samples).clip(6, 180)
        
        return pd.DataFrame({
            'income_stability_index': income_stability_index,
            'employment_tenure_months': employment_tenure_months,
            'bank_account_age_months': bank_account_age_months
        })
    
    def create_target_variables(self, features_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Create target variables based on affordability logic.
        
        Args:
            features_df: DataFrame with all synthesized features
            
        Returns:
            Tuple of (Repayment_Status, Max_Safe_Loan_Amount)
            
        Note: Standard classification labels
            - 0 = Repays (Positive outcome)
            - 1 = Defaults (Negative outcome)
        """
        # Repayment Status Logic - using more realistic thresholds
        # Discipline: Good payment history
        discipline_score = (
            (features_df['on_time_rent_payments_pct'] > 85) & 
            (features_df['utility_payment_consistency'] > 80) &
            (features_df['active_subscription_months'] > 6)
        ).astype(int)
        
        # Capacity: Manageable debt and reasonable income
        capacity_score = (
            (features_df['alt_debt_to_income_ratio'] < 0.6) &
            (features_df['monthly_net_income'] > 2000)
        ).astype(int)
        
        # Stability: Consistent income and employment
        stability_score = (
            (features_df['income_stability_index'] > 3.5) &
            (features_df['employment_tenure_months'] > 3)
        ).astype(int)
        
        # Calculate composite score (0-3)
        composite_score = discipline_score + capacity_score + stability_score
        
        # Default probability based on composite score (INVERTED LOGIC)
        # Score 3: 5% default, Score 2: 30% default, Score 1: 70% default, Score 0: 95% default
        noise = np.random.random(len(features_df))
        repayment_status = np.where(
            composite_score == 3, (noise < 0.05).astype(int),  # 5% default
            np.where(composite_score == 2, (noise < 0.30).astype(int),  # 30% default
            np.where(composite_score == 1, (noise < 0.70).astype(int),  # 70% default
            (noise < 0.95).astype(int)))  # 95% default
        )
        
        # Max Safe Liquidity Access (Subscription Model):
        # This is NOT a loan - it's a subscription for cash access (like Netflix for money)
        # Borrowing capacity = 15% of 3-month income (collateral limit)
        # Subscription fee = $16.12/month (competes with Netflix)
        
        # Calculate 3-month income window
        three_month_income = features_df['monthly_net_income'] * 3
        
        # Max borrowing capacity: 15% of 3-month income
        borrowing_capacity_rate = 0.15
        max_borrowing_capacity = three_month_income * borrowing_capacity_rate
        
        # Disposable income check (ensure they can afford subscription after essentials)
        monthly_essentials = (
            features_df['monthly_rent'] + 
            features_df['monthly_recurring_bills'] + 
            500  # buffer for food
        )
        disposable_monthly = (features_df['monthly_net_income'] - monthly_essentials).clip(0, None)
        
        # Apply realistic borrowing range with safety caps
        # Lower end: $300, Upper end: min(borrowing_capacity, $3000)
        max_safe_loan_amount = np.random.uniform(0.5, 1.0, len(features_df)) * max_borrowing_capacity
        max_safe_loan_amount = max_safe_loan_amount.clip(300, 3000)
        
        # Additional safety: ensure they have disposable income
        # Must have at least $200/month disposable to qualify
        max_safe_loan_amount = np.where(disposable_monthly < 200, 0, max_safe_loan_amount)
        
        # Calculate subscription fee (appears in output for reference)
        # Subscription is constant: $16.12/month (psychological pricing)
        subscription_fee = 16.12  # Netflix competitor pricing
        
        return pd.Series(repayment_status, name='Repayment_Status'), pd.Series(max_safe_loan_amount, name='Max_Safe_Loan_Amount')
    
    def generate_complete_dataset(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate complete synthetic dataset with all features and targets.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Complete DataFrame ready for modeling
        """
        print(f"Generating {n_samples} synthetic samples...")
        
        # Generate all feature categories
        discipline_df = self.synthesize_discipline_features(n_samples)
        capacity_df = self.synthesize_capacity_features(n_samples)
        stability_df = self.synthesize_stability_features(
            n_samples, 
            capacity_df['monthly_net_income'].values
        )
        
        # Combine all features
        features_df = pd.concat([discipline_df, capacity_df, stability_df], axis=1)
        
        # Create target variables
        repayment_status, max_safe_loan = self.create_target_variables(features_df)
        
        # Add targets to dataset
        complete_df = features_df.copy()
        complete_df['Repayment_Status'] = repayment_status
        complete_df['Max_Safe_Loan_Amount'] = max_safe_loan
        complete_df['Monthly_Subscription_Fee'] = 16.12  # Constant subscription fee
        
        print(f"Dataset generated successfully!")
        print(f"Default rate: {repayment_status.mean():.2%}")
        print(f"Repayment rate: {(1 - repayment_status.mean()):.2%}")
        print(f"Average safe loan amount: ${max_safe_loan.mean():.2f}")
        print(f"Monthly subscription: $16.12 (Netflix competitor pricing)")
        
        return complete_df


def save_synthetic_data(output_path: str = 'data/synthetic/synthetic_credit_data.csv', n_samples: int = 5000000):
    """
    Generate and save synthetic dataset to CSV.
    
    Args:
        output_path: Path to save the CSV file
        n_samples: Number of samples to generate
    """
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate data
    synthesizer = AlternativeCreditSynthesizer(seed=42)
    df = synthesizer.generate_complete_dataset(n_samples)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate and save dataset when run directly (default: 5M samples)
    df = save_synthetic_data(n_samples=5000000)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
