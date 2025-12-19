"""
Reverse Savings Credit System - Week 1
Data Acquisition and Synthesis Strategy
"""

from Library.data_synthesizer import AlternativeCreditSynthesizer, save_synthetic_data
from Library.api_integrations import AlternativeCreditDataAggregator
from Library.model_evaluation import ConfusionMatrixAnalyzer
import pandas as pd
import os
import sys
import numpy as np # Added import for numpy

def week1_setup(use_spark=False, n_samples=5000000):
    """
    Week 1: Environment Setup and Data Synthesis
    """
    print("=" * 60)
    print("REVERSE SAVINGS CREDIT SYSTEM")
    print("Setup & Data Acquisition")
    print("=" * 60)
    print()

    # Step 1: Generate synthetic alternative credit features
    print("Step 1: Generating Synthetic Alternative Credit Features")
    print("-" * 60)

    # Ensure the data directory exists
    os.makedirs('data/synthetic', exist_ok=True)

    # Always use consistent naming with sample size suffix
    output_path = f'data/synthetic/synthetic_credit_data_{n_samples//1000}K.csv'
    
    if use_spark:
        try:
            from Library.pyspark_data_generator import generate_spark_data
            generate_spark_data(n_samples=n_samples, output_path=output_path)
            df = pd.read_csv(output_path)
        except (ImportError, ValueError) as e:
            print(f"\n⚠ Spark not available: {e}")
            print("✓ Falling back to standard Python generator (works well for <1M samples)\n")
            df = save_synthetic_data(output_path=output_path, n_samples=n_samples)
    else:
        df = save_synthetic_data(output_path=output_path, n_samples=n_samples)

    # Step 2: Display dataset summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"\nTotal samples: {len(df)}")
    print(f"\nFeature columns: {len(df.columns) - 2}")  # excluding targets
    print(f"Target variables: 2 (Repayment_Status, Max_Safe_Loan_Amount)")

    print("\n--- Discipline Features ---")
    print(df[['on_time_rent_payments_pct', 'active_subscription_months', 'utility_payment_consistency']].describe())

    print("\n--- Capacity Features ---")
    print(df[['monthly_net_income', 'paycheck_amount', 'pay_frequency', 'alt_debt_to_income_ratio']].describe())
    print("\nPay Frequency Distribution:")
    print(df['pay_frequency'].value_counts())

    print("\n--- Stability Features ---")
    print(df[['income_stability_index', 'employment_tenure_months']].describe())

    print("\n--- Target Variables ---")
    print(f"Repayment Status Distribution:")
    print(df['Repayment_Status'].value_counts())
    print(f"\nRepayment Rate: {df['Repayment_Status'].mean():.2%}")
    print(f"\nMax Safe Loan Amount:")
    print(df['Max_Safe_Loan_Amount'].describe())
    print(f"\n--- Subscription Model ---")
    print(f"Monthly Subscription Fee: ${df['Monthly_Subscription_Fee'].iloc[0]:.2f}")
    print(f"Borrowing Capacity: 15% of 3-month income")
    print(f"Example: $2,000/month → ${2000*3*0.15:.0f} max borrowing")

    print("\n" + "=" * 60)
    print("Data Acquisition Complete! ✓")
    print("Next: Data Preparation & EDA")
    print("=" * 60)


def create_ml_outcome_summary(df):
    """
    Creates a summary of ML data outcomes and saves it to a file.
    """
    summary_data = {
        "Total Samples": len(df),
        "Repayment Rate (%)": df['Repayment_Status'].mean() * 100,
        "Average Max Safe Loan Amount": df['Max_Safe_Loan_Amount'].mean(),
        "Feature Count": len(df.columns) - 2,
        "Target Count": 2
    }
    summary_df = pd.DataFrame([summary_data])

    # Ensure the summaries directory exists
    os.makedirs('data/summaries', exist_ok=True)
    summary_path = 'data/summaries/ml_outcomes_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("--- Machine Learning Data Outcomes Summary ---\n\n")
        for key, value in summary_data.items():
            f.write(f"{key}: {value:.2f}\n")
        f.write("\n--- End of Summary ---")
    print(f"\nML outcomes summary saved to: {summary_path}")


def run_all_weeks():
    """Run all weeks sequentially."""
    print("\n" + "=" * 70)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 70 + "\n")

    import os
    import glob

    # Delete any existing cache files
    cache_files = glob.glob('data/models_output/*.npz')
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
        except:
            pass

    # Ask for sample size
    print("\nHow much data would you like to generate?")
    print("Enter a percentage (1-100) of the full 1M sample dataset.")
    print("Examples:")
    print("  - 10  = 100,000 samples (~1-2 min)")
    print("  - 50  = 500,000 samples (~5-7 min)")
    print("  - 100 = Full 1,000,000 samples (DEFAULT, recommended, ~10-12 min)")

    percentage_confirmed = False
    fresh_data_percentage = 1.0  # Default to 100% = 1M samples
    samples_to_generate = 1000000  # Default to 1M

    while not percentage_confirmed:
        percentage_input = input("\nEnter percentage (1-100): ").strip()
        try:
            percentage = float(percentage_input)
            if 0 < percentage <= 100:
                fresh_data_percentage = percentage / 100.0
                samples_to_generate = int(1000000 * fresh_data_percentage)

                # Ensure minimum samples
                min_samples = 10000
                if samples_to_generate < min_samples:
                    print(f"Warning: Adjusting to minimum {min_samples} samples")
                    samples_to_generate = min_samples

                print(f"\n📋 You selected: {percentage}%")
                print(f"   This will generate {samples_to_generate:,} samples")
                if percentage >= 75:
                    print(f"   ⏱ Estimated time: 10-12 minutes")
                elif percentage >= 25:
                    print(f"   ⏱ Estimated time: 5-7 minutes")
                else:
                    print(f"   ⏱ Estimated time: < 2 minutes")

                confirm_pct = input("\nConfirm this percentage? (yes/no): ").strip().lower()
                if confirm_pct in ['yes', 'y', 'ye', 'yse', 'ys']:
                    print(f"\n✓ Confirmed - will generate {samples_to_generate:,} samples")
                    percentage_confirmed = True
                else:
                    print(f"\n❌ Not confirmed - let's try again...\n")
            else:
                print("\n⚠ Invalid percentage. Please enter a value between 1 and 100.")
        except ValueError:
            print("\n⚠ Invalid input. Please enter a number (e.g., 50).")

    print()

    # Week 1: Data Generation
    print(f"\n🚀 Generating {samples_to_generate:,} samples ({fresh_data_percentage*100:.0f}% of 1M)...\n")
    print("ℹ️  Using standard Python generator (no Spark/Java required)\n")
    # Always use standard Python - no Spark dependency
    week1_setup(use_spark=False, n_samples=samples_to_generate)

    # Load synthetic data for subsequent weeks
    df = None
    try:
        # The file was just generated with the sample size we requested
        generated_file = f'data/synthetic/synthetic_credit_data_{samples_to_generate//1000}K.csv'

        if os.path.exists(generated_file):
            df = pd.read_csv(generated_file)
            actual_samples = len(df)
            print(f"✓ Loaded fresh data from: {generated_file}")
            print(f"✓ Confirmed sample count: {actual_samples:,} samples")

            # Verify we got what we asked for
            if actual_samples != samples_to_generate:
                print(f"⚠ WARNING: Expected {samples_to_generate:,} but got {actual_samples:,}")
        else:
            raise FileNotFoundError(f"Generated file not found: {generated_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure Week 1 setup completed successfully.")
        return

    # Create ML outcomes summary
    create_ml_outcome_summary(df)

    # Week 2
    print("\n")
    from Library import data_preparation
    data_preparation.week2_main()

    # Week 3
    print("\n")
    from Library import core_modeling
    core_modeling.week3_main()

    # Generate comprehensive model statistics
    from Library.model_statistics import generate_model_statistics
    generate_model_statistics()

    # Week 4: Regression Modeling
    print("\n" + "="*70)
    print("WEEK 4: REGRESSION MODELING")
    print("="*70)
    from Library.regression_modeling import regression_main
    regression_main()

    # Week 4.5: Batched Cost Analysis (for report)
    print("\n" + "="*70)
    print("WEEK 4.5: BATCHED COST ANALYSIS")
    print("="*70)
    from Library.batched_cost_analysis import run_batched_cost_analysis

    # Load processed data to get predictions
    processed_data = np.load('data/models_output/processed_data.npz', allow_pickle=True)

    # Use CLASSIFICATION test data (20 features), not regression data (15 features)
    X_test_class = processed_data['X_test']  # Classification features (20 features)
    y_class_test = processed_data['y_class_test']  # Classification target

    # Load the primary CLASSIFICATION XGBoost model from Week 3
    import joblib
    import os

    # Use the primary XGBoost classification model
    if os.path.exists('models/xgboost_primary.pkl'):
        xgb_model = joblib.load('models/xgboost_primary.pkl')
        print("✓ Using primary XGBoost classification model")
    elif os.path.exists('models/optimized/xgboost_optimized.pkl'):
        xgb_model = joblib.load('models/optimized/xgboost_optimized.pkl')
        print("✓ Using optimized XGBoost classification model")
    else:
        print("⚠ No XGBoost classification model found, skipping batched cost analysis")
        xgb_model = None

    if xgb_model is not None:
        # Make predictions with the classification model (expects 20 features)
        y_pred = xgb_model.predict(X_test_class)

        # Run batched analysis
        run_batched_cost_analysis(
            y_class_test,
            y_pred,
            batch_size=100,
            threshold=2,
            fn_weight=10,
            fp_weight=1
        )

    # Week 5: SHAP Explainability
    from Library.shap_explainability import week5_main as shap_week5
    shap_week5()

    # Week 6: Advanced Optimization (if applicable - not explicitly defined here, assuming it's part of core_modeling or separate)
    # Placeholder for any potential Week 6 activities if they exist and are not covered by 'all'

    # Week 7: Monte Carlo Simulation
    if 'all' in sys.argv or 'monte_carlo' in sys.argv:
        print_week_header(7, "Monte Carlo Portfolio Risk Simulation")
        from Library.monte_carlo_simulation import run_monte_carlo_analysis
        run_monte_carlo_analysis()
        print()

    # Threshold Stress Test
    if 'all' in sys.argv or 'stress_test' in sys.argv:
        print_week_header(8, "Threshold Stress Test Analysis")
        from Library.threshold_stress_test import main as stress_test_main
        stress_test_main()
        print()

    print_final_summary()


def print_week_header(week_num, title):
    """Prints a formatted header for each week."""
    print("\n" + "=" * 70)
    print(f"WEEK {week_num}: {title}")
    print("=" * 70)

def print_final_summary():
    """Prints the final summary message."""
    print("\n" + "*" * 70)
    print("COMPLETE PIPELINE EXECUTION FINISHED")
    print("*" * 70)


if __name__ == "__main__":
    import sys

    # Helper to get workflow argument, defaults to 'all'
    def get_workflow():
        if len(sys.argv) > 1:
            return sys.argv[1].lower()
        return 'all'

    workflow = get_workflow()

    # --- MAPPING COMMANDS TO WORKFLOWS ---
    # This part maps the command-line arguments to specific actions or workflows.
    # The original code had a long list of elifs. Consolidating them into a more
    # readable structure or mapping is a good practice.
    # For this edit, we are focusing on adding 'stress_test' to the 'all' workflow.

    # Explicitly handle 'all' to run all defined weeks
    if workflow == "all":
        run_all_weeks()

    # Handle individual week runs or specific tasks
    elif workflow == "1" or workflow == "week1":
        print("\n" + "=" * 70)
        print("RUNNING WEEK 1: DATA SYNTHESIS")
        print("=" * 70 + "\n")
        week1_setup(use_spark=True, n_samples=1000000) # Default to full 1M for standalone week1
    elif workflow == "spark":
        print("\n" + "=" * 70)
        print("RUNNING WEEK 1 (SPARK): 1M SAMPLES")
        print("=" * 70 + "\n")
        week1_setup(use_spark=True, n_samples=1000000)
    elif workflow == "spark_test":
        print("\n" + "=" * 70)
        print("RUNNING WEEK 1 (SPARK TEST): 100K SAMPLES")
        print("=" * 70 + "\n")
        week1_setup(use_spark=True, n_samples=100000)
    elif workflow == "2" or workflow == "week2":
        print("\n" + "=" * 70)
        print("RUNNING WEEK 2: DATA PREPARATION & EDA")
        print("=" * 70 + "\n")
        # Need to ensure data is available before running week 2
        # For simplicity, we'll assume data is either cached or generated by previous steps
        # A more robust script would handle data loading/generation here if needed.
        try:
            from Library import data_preparation
            data_preparation.week2_main()
        except Exception as e:
            print(f"Error running Week 2: {e}")
            print("Please ensure Week 1 has been successfully completed or data is available.")
    elif workflow == "3" or workflow == "week3":
        print("\n" + "=" * 70)
        print("RUNNING WEEK 3: CORE MODELING & STATISTICS")
        print("=" * 70 + "\n")
        try:
            from Library import core_modeling
            core_modeling.week3_main()
            from Library import model_statistics
            model_statistics.generate_model_statistics()
        except Exception as e:
            print(f"Error running Week 3: {e}")
            print("Please ensure Week 1 and Week 2 have been successfully completed.")
    # Added elif for 'regression' to run only regression modeling
    elif workflow == "regression" or workflow == "week4":
        print("\n" + "=" * 70)
        print("RUNNING WEEK 4: REGRESSION MODELING")
        print("=" * 70 + "\n")
        try:
            from Library.regression_modeling import regression_main
            regression_main()
        except Exception as e:
            print(f"Error running Regression Modeling: {e}")
            print("Please ensure previous weeks have been completed.")
    # Added elif for 'optimize' to run advanced optimization
    elif workflow == "optimize":
        print("\n" + "=" * 70)
        print("RUNNING ADVANCED OPTIMIZATION")
        print("=" * 70 + "\n")
        try:
            from Library.advanced_optimization import run_advanced_optimization
            run_advanced_optimization()
        except Exception as e:
            print(f"Error running Advanced Optimization: {e}")
    # Added 'stats' command for generating all statistics
    elif workflow == "stats":
        print("\n" + "=" * 70)
        print("GENERATING ALL MODEL STATISTICS")
        print("=" * 70 + "\n")
        try:
            from Library.model_statistics import generate_all_statistics
            generate_all_statistics()
        except Exception as e:
            print(f"Error generating statistics: {e}")
    # Added 'monte-carlo' command
    elif workflow == "monte-carlo" or workflow == "monte_carlo":
        print_week_header(7, "Monte Carlo Portfolio Risk Simulation")
        try:
            from Library.monte_carlo_simulation import run_monte_carlo_analysis
            run_monte_carlo_analysis()
        except Exception as e:
            print(f"Error running Monte Carlo simulation: {e}")
    # Added the new 'stress_test' command
    elif workflow == "stress_test":
        print("\n" + "=" * 70)
        print("RUNNING THRESHOLD STRESS TEST")
        print("=" * 70 + "\n")
        try:
            from Library.threshold_stress_test import main as stress_test_main
            stress_test_main()
        except Exception as e:
            print(f"Error running Threshold Stress Test: {e}")

    else:
        print(f"Unknown command: {workflow}")
        print("\nAvailable commands:")
        print("  python main.py all              - Run complete pipeline")
        print("  python main.py week1            - Week 1: Data synthesis")
        print("  python main.py week2            - Week 2: Data preparation & EDA")
        print("  python main.py week3            - Week 3: Core modeling")
        print("  python main.py regression       - Week 4: Regression modeling")
        print("  python main.py optimize         - Advanced model optimization")
        print("  python main.py stats            - Generate model statistics")
        print("  python main.py monte-carlo      - Run Monte Carlo simulation")
        print("  python main.py stress_test      - Run Threshold Stress Test")