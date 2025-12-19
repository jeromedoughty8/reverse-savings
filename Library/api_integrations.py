
"""
API Integration Module for Alternative Credit Data
Integrates with Plaid, Experian Boost, and Open Banking APIs
"""

import os
from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta


class PlaidIntegration:
    """
    Integration with Plaid API for transaction and income data.
    Docs: https://plaid.com/docs/
    """
    
    def __init__(self, client_id: str = None, secret: str = None):
        """Initialize Plaid client with credentials from environment."""
        self.client_id = client_id or os.getenv('PLAID_CLIENT_ID')
        self.secret = secret or os.getenv('PLAID_SECRET')
        # In production: from plaid import Client
        # self.client = Client(client_id=self.client_id, secret=self.secret, environment='sandbox')
    
    def get_transaction_data(self, access_token: str, days: int = 90) -> pd.DataFrame:
        """
        Fetch transaction history for discipline and capacity features.
        
        Args:
            access_token: User's Plaid access token
            days: Number of days of history to fetch
            
        Returns:
            DataFrame with transactions
        """
        # Production implementation:
        # start_date = (datetime.now() - timedelta(days=days)).date()
        # end_date = datetime.now().date()
        # response = self.client.Transactions.get(access_token, start_date, end_date)
        # transactions = response['transactions']
        
        # For now, return structure showing what we'd extract:
        print(f"[Plaid] Fetching {days} days of transaction data...")
        return pd.DataFrame({
            'date': [],
            'amount': [],
            'category': [],  # e.g., 'Rent', 'Utilities', 'Subscriptions'
            'merchant_name': []
        })
    
    def get_income_data(self, access_token: str) -> Dict:
        """
        Fetch income verification data for capacity features.
        
        Returns:
            Dict with income details including stability metrics
        """
        # Production implementation:
        # response = self.client.Income.get(access_token)
        # income_streams = response['income']['income_streams']
        
        print("[Plaid] Fetching income verification data...")
        return {
            'monthly_net_income': 0.0,
            'income_sources': [],
            'pay_frequency': '',  # 'WEEKLY', 'BIWEEKLY', 'MONTHLY'
            'last_6_months_avg': 0.0,
            'income_variance': 0.0  # For stability index
        }
    
    def calculate_discipline_features(self, transactions_df: pd.DataFrame) -> Dict:
        """
        Extract discipline features from transaction history.
        
        Returns:
            Dict with rent payment consistency, utility payments, etc.
        """
        # Filter rent payments (Plaid categorizes these)
        rent_txns = transactions_df[transactions_df['category'].str.contains('Rent', na=False)]
        
        # Calculate on-time rent payment percentage
        # (Would need to compare against due dates in production)
        on_time_rent_pct = 0.0  # Placeholder
        
        # Find recurring subscriptions
        subscription_txns = transactions_df[
            transactions_df['category'].str.contains('Subscription', na=False)
        ]
        active_subscription_months = len(subscription_txns['date'].dt.to_period('M').unique())
        
        # Utility payment consistency
        utility_txns = transactions_df[
            transactions_df['category'].str.contains('Utilities', na=False)
        ]
        utility_consistency = 0.0  # Placeholder
        
        return {
            'on_time_rent_payments_pct': on_time_rent_pct,
            'active_subscription_months': active_subscription_months,
            'utility_payment_consistency': utility_consistency
        }


class ExperianBoostIntegration:
    """
    Integration with Experian Boost for utility and telecom payment history.
    Docs: https://developer.experian.com/
    """
    
    def __init__(self, api_key: str = None):
        """Initialize Experian API client."""
        self.api_key = api_key or os.getenv('EXPERIAN_API_KEY')
        # In production: configure OAuth or API key authentication
    
    def get_utility_payment_history(self, user_id: str) -> Dict:
        """
        Fetch utility and telecom payment history.
        
        Returns:
            Dict with payment history metrics
        """
        # Production implementation would call Experian Boost API
        print("[Experian Boost] Fetching utility payment history...")
        
        return {
            'utility_accounts': [],  # List of utility accounts
            'on_time_payment_pct': 0.0,
            'total_accounts': 0,
            'oldest_account_months': 0,
            'recent_missed_payments': 0
        }
    
    def get_telecom_payment_history(self, user_id: str) -> Dict:
        """
        Fetch telecom (phone, internet) payment history.
        
        Returns:
            Dict with telecom payment metrics
        """
        print("[Experian Boost] Fetching telecom payment history...")
        
        return {
            'telecom_accounts': [],
            'on_time_payment_pct': 0.0,
            'account_age_months': 0
        }


class OpenBankingIntegration:
    """
    Integration with Open Banking APIs (e.g., UK Open Banking, PSD2 in EU).
    Provides direct access to bank account data with user consent.
    """
    
    def __init__(self, api_key: str = None, environment: str = 'sandbox'):
        """Initialize Open Banking client."""
        self.api_key = api_key or os.getenv('OPEN_BANKING_API_KEY')
        self.environment = environment
        # In production: use provider like TrueLayer, Plaid (EU), or Tink
    
    def get_account_balance(self, account_id: str) -> Dict:
        """
        Fetch current account balance and available funds.
        
        Returns:
            Dict with balance information
        """
        print("[Open Banking] Fetching account balance...")
        
        return {
            'current_balance': 0.0,
            'available_balance': 0.0,
            'currency': 'USD',
            'account_type': ''  # 'CHECKING', 'SAVINGS'
        }
    
    def get_direct_debits(self, account_id: str) -> pd.DataFrame:
        """
        Fetch recurring direct debits (subscriptions, bills).
        
        Returns:
            DataFrame with recurring payment obligations
        """
        print("[Open Banking] Fetching direct debits...")
        
        return pd.DataFrame({
            'merchant': [],
            'amount': [],
            'frequency': [],  # 'MONTHLY', 'WEEKLY'
            'next_payment_date': []
        })
    
    def calculate_capacity_features(self, account_id: str, days: int = 90) -> Dict:
        """
        Calculate capacity features from bank account data.
        
        Returns:
            Dict with income, DTI, and stability metrics
        """
        # Fetch transactions to identify income sources
        print("[Open Banking] Analyzing account for capacity metrics...")
        
        # In production: analyze direct deposits, recurring expenses
        return {
            'monthly_net_income': 0.0,
            'monthly_rent': 0.0,
            'monthly_recurring_bills': 0.0,
            'alt_debt_to_income_ratio': 0.0,
            'income_stability_index': 0.0,
            'bank_account_age_months': 0
        }


class AlternativeCreditDataAggregator:
    """
    Orchestrates all API integrations to build complete alternative credit profile.
    """
    
    def __init__(self):
        """Initialize all API clients."""
        self.plaid = PlaidIntegration()
        self.experian = ExperianBoostIntegration()
        self.open_banking = OpenBankingIntegration()
    
    def build_credit_profile(self, 
                            plaid_access_token: Optional[str] = None,
                            experian_user_id: Optional[str] = None,
                            open_banking_account_id: Optional[str] = None) -> Dict:
        """
        Aggregate data from all sources to build complete alternative credit profile.
        
        Args:
            plaid_access_token: User's Plaid access token (if linked)
            experian_user_id: Experian user ID (if enrolled in Boost)
            open_banking_account_id: Open Banking account ID (if authorized)
            
        Returns:
            Dict with all alternative credit features ready for model input
        """
        profile = {}
        
        # Plaid data (if available)
        if plaid_access_token:
            transactions = self.plaid.get_transaction_data(plaid_access_token)
            income_data = self.plaid.get_income_data(plaid_access_token)
            discipline_features = self.plaid.calculate_discipline_features(transactions)
            
            profile.update(discipline_features)
            profile['monthly_net_income'] = income_data['monthly_net_income']
            profile['income_stability_index'] = 1 / (income_data['income_variance'] + 0.01)
        
        # Experian Boost data (if available)
        if experian_user_id:
            utility_data = self.experian.get_utility_payment_history(experian_user_id)
            telecom_data = self.experian.get_telecom_payment_history(experian_user_id)
            
            # Blend with Plaid data or use as primary source
            if 'utility_payment_consistency' not in profile:
                profile['utility_payment_consistency'] = utility_data['on_time_payment_pct']
        
        # Open Banking data (if available)
        if open_banking_account_id:
            capacity_data = self.open_banking.calculate_capacity_features(open_banking_account_id)
            direct_debits = self.open_banking.get_direct_debits(open_banking_account_id)
            
            # Use as alternative or validation for Plaid data
            profile.update(capacity_data)
        
        # Fill in any missing features with defaults or reject application
        required_features = [
            'on_time_rent_payments_pct',
            'active_subscription_months',
            'utility_payment_consistency',
            'monthly_net_income',
            'alt_debt_to_income_ratio',
            'income_stability_index'
        ]
        
        for feature in required_features:
            if feature not in profile:
                profile[feature] = None  # Mark as missing data
        
        return profile


# Example usage
if __name__ == "__main__":
    print("=== Alternative Credit Data API Integration ===\n")
    
    # Initialize aggregator
    aggregator = AlternativeCreditDataAggregator()
    
    # In production, these tokens would come from user authentication flow
    profile = aggregator.build_credit_profile(
        plaid_access_token="access-sandbox-xxx",  # From Plaid Link flow
        experian_user_id="user-123",
        open_banking_account_id="account-456"
    )
    
    print("\nExtracted Alternative Credit Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    
    print("\n=== Integration Setup Complete ===")
    print("\nNext Steps:")
    print("1. Set up API credentials in Replit Secrets:")
    print("   - PLAID_CLIENT_ID")
    print("   - PLAID_SECRET")
    print("   - EXPERIAN_API_KEY")
    print("   - OPEN_BANKING_API_KEY")
    print("2. Implement OAuth flows for user consent")
    print("3. Replace synthetic data in data_synthesizer.py with API calls")
