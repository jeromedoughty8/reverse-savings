"""
Reverse Savings Credit System - Week 2
Data Preparation & Exploratory Data Analysis (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_inspect_data(file_path='data/synthetic/synthetic_credit_data.csv'):
    """Load the synthetic dataset and perform initial inspection."""
    print("=" * 70)
    print("DATA PREPARATION & EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print()

    print("Step 1: Loading Dataset")
    print("-" * 70)
    df = pd.read_csv(file_path)
    print(f"✓ Dataset loaded: {len(df):,} samples, {len(df.columns)} features")
    print()

    # Check for missing values
    print("Step 2: Data Quality Check")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found")
    else:
        print(f"⚠ Missing values detected:\n{missing[missing > 0]}")
    print()

    # Data types
    print("Data Types:")
    print(df.dtypes)
    print()

    return df

def prepare_features(df):
    """Prepare features for modeling: encoding and scaling."""
    print("Step 3: Feature Engineering")
    print("-" * 70)

    # One-hot encode pay_frequency (categorical variable)
    print("Encoding categorical variable: pay_frequency")
    df_encoded = pd.get_dummies(df, columns=['pay_frequency'], prefix='pay_freq', drop_first=False)
    print(f"✓ Created dummy variables: {[col for col in df_encoded.columns if 'pay_freq' in col]}")
    print()

    # Separate features and targets
    target_cols = ['Repayment_Status', 'Max_Safe_Loan_Amount', 'Monthly_Subscription_Fee']
    feature_cols = [col for col in df_encoded.columns if col not in target_cols]

    X = df_encoded[feature_cols]
    y_classification = df_encoded['Repayment_Status']
    y_regression = df_encoded['Max_Safe_Loan_Amount']

    print(f"Features: {len(feature_cols)} columns")
    print(f"Classification target: Repayment_Status (Binary: 0=Repays, 1=Defaults)")
    print(f"Regression target: Max_Safe_Loan_Amount (Continuous: ${y_regression.min():.0f} - ${y_regression.max():.0f})")
    print()

    return X, y_classification, y_regression, df_encoded

def exploratory_data_analysis(df, output_dir='data/eda_plots'):
    """Perform comprehensive EDA with visualizations."""
    print("Step 4: Exploratory Data Analysis")
    print("-" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Target variable distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Repayment Status
    repayment_counts = df['Repayment_Status'].value_counts()
    axes[0].bar(['Repays (0)', 'Defaults (1)'], repayment_counts.values, color=['green', 'red'], alpha=0.7)
    axes[0].set_title('Repayment Status Distribution', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Count')
    axes[0].text(0, repayment_counts[0], f'{repayment_counts[0]:,}\n({repayment_counts[0]/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')
    axes[0].text(1, repayment_counts[1], f'{repayment_counts[1]:,}\n({repayment_counts[1]/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    # Max Safe Loan Amount
    axes[1].hist(df['Max_Safe_Loan_Amount'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_title('Max Safe Loan Amount Distribution', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Loan Amount ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(df['Max_Safe_Loan_Amount'].mean(), color='red', linestyle='--',
                    label=f'Mean: ${df["Max_Safe_Loan_Amount"].mean():.0f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_target_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/01_target_distributions.png")
    plt.close()

    # 2. Feature correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Monthly_Subscription_Fee']]

    plt.figure(figsize=(14, 10))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/02_correlation_matrix.png")
    plt.close()

    # 3. Key feature distributions by repayment status
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Feature Distributions by Repayment Status', fontweight='bold', fontsize=14)

    key_features = [
        'on_time_rent_payments_pct',
        'monthly_net_income',
        'alt_debt_to_income_ratio',
        'income_stability_index',
        'employment_tenure_months',
        'active_subscription_months'
    ]

    for idx, feature in enumerate(key_features):
        row = idx // 3
        col = idx % 3

        df[df['Repayment_Status'] == 0][feature].hist(ax=axes[row, col], bins=30,
                                                        alpha=0.6, label='Repays', color='green', edgecolor='black')
        df[df['Repayment_Status'] == 1][feature].hist(ax=axes[row, col], bins=30,
                                                        alpha=0.6, label='Defaults', color='red', edgecolor='black')
        axes[row, col].set_title(feature.replace('_', ' ').title(), fontsize=10)
        axes[row, col].legend()
        axes[row, col].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/03_feature_distributions.png")
    plt.close()

    # 4. Income vs Loan Amount scatter
    plt.figure(figsize=(10, 6))
    colors = ['green' if x == 0 else 'red' for x in df['Repayment_Status']]
    plt.scatter(df['monthly_net_income'], df['Max_Safe_Loan_Amount'],
                c=colors, alpha=0.5, s=20)
    plt.xlabel('Monthly Net Income ($)', fontweight='bold')
    plt.ylabel('Max Safe Loan Amount ($)', fontweight='bold')
    plt.title('Income vs Borrowing Capacity (colored by Repayment Status)', fontweight='bold')
    plt.grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='Repays'),
                      Patch(facecolor='red', alpha=0.5, label='Defaults')]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_income_vs_loan.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/04_income_vs_loan.png")
    plt.close()

    print()
    print("EDA Summary Statistics:")
    print("-" * 70)
    print(f"Default Rate: {df['Repayment_Status'].mean():.2%}")
    print(f"Average Loan Amount (All): ${df['Max_Safe_Loan_Amount'].mean():.2f}")
    print(f"Average Loan Amount (Repays): ${df[df['Repayment_Status']==0]['Max_Safe_Loan_Amount'].mean():.2f}")
    print(f"Average Loan Amount (Defaults): ${df[df['Repayment_Status']==1]['Max_Safe_Loan_Amount'].mean():.2f}")
    print()

def create_train_test_split(X, y_classification, y_regression, test_size=0.2, random_state=42):
    """Create train/test splits for both classification and regression tasks."""
    print("Step 5: Train/Test Split")
    print("-" * 70)

    # Split data (70/30 train/test for large dataset - default 5M samples)
    # With 5,000,000 samples, 30% test = 1,500,000 samples for robust evaluation
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.3, random_state=random_state, stratify=y_classification
    )

    # Split regression targets using same indices
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.3, random_state=random_state, stratify=y_classification
    )

    print(f"✓ Train set: {len(X_train)} samples (70%)")
    print(f"✓ Test set: {len(X_test)} samples (30%)")
    print(f"  - Train default rate: {y_class_train.mean():.2%}")
    print(f"  - Test default rate: {y_class_test.mean():.2%}")
    print()

    # Feature scaling (important for some models)
    print("Step 6: Feature Scaling")
    print("-" * 70)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features standardized (mean=0, std=1)")
    print()

    # Always save processed data (cache for next run)
    os.makedirs('data/models_output', exist_ok=True)

    # Delete any existing cache files first to ensure we save the most recent
    import glob
    old_cache_files = glob.glob('data/models_output/processed_data*.npz')
    for old_file in old_cache_files:
        try:
            os.remove(old_file)
            print(f"  └─ Removed old cache: {old_file}")
        except:
            pass

    total_samples = len(X_train) + len(X_test)
    output_path = 'data/models_output/processed_data.npz'

    print("\n" + "=" * 70)
    print(f"SAVING PROCESSED DATA TO CACHE")
    print("=" * 70)
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Total samples: {total_samples:,}")
    print(f"Target path: {output_path}")

    np.savez(
        output_path,
             X_train=X_train.values, X_test=X_test.values,
             X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
             y_class_train=y_class_train.values, y_class_test=y_class_test.values,
             y_reg_train=y_reg_train.values, y_reg_test=y_reg_test.values,
             feature_names=X_train.columns.values,
             total_samples=np.array([total_samples]))  # Store sample count in cache
    print(f"✓ Processed data saved to: {output_path}")
    print(f"  └─ Cache contains {total_samples:,} samples and will be used in next run")
    print()

    return X_train, X_test, X_train_scaled, X_test_scaled, y_class_train, y_class_test, y_reg_train, y_reg_test, scaler

def week2_main():
    """Execute Week 2 workflow."""
    # Load data
    df = load_and_inspect_data()

    # If user chose to use cached data, skip processing
    if df is None:
        print("\n" + "=" * 70)
        print("WEEK 2 SKIPPED (Using Cached Data)")
        print("=" * 70)
        print("Proceeding with existing processed_data.npz")
        print()
        return

    # Prepare features
    X, y_classification, y_regression, df_encoded = prepare_features(df)

    # EDA
    exploratory_data_analysis(df)

    # Train/test split
    splits = create_train_test_split(X, y_classification, y_regression)

    print("=" * 70)
    print("DATA PREPARATION COMPLETE! ✓")
    print("=" * 70)
    print("Deliverables:")
    print("  ✓ Data cleaned and encoded")
    print("  ✓ EDA visualizations generated (4 plots)")
    print("  ✓ Train/test split created (80/20)")
    print("  ✓ Features scaled and ready for modeling")
    print()
    print("Next: Core Modeling & Baseline")
    print("=" * 70)

if __name__ == "__main__":
    week2_main()