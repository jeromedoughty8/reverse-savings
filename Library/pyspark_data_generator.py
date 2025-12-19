"""
PySpark Data Generator with Credit Risk Metrics
Generates 1M samples with PD, LGD, EAD, RR, ECL, and ALLL
"""

import os
import subprocess

# Set JAVA_HOME if not already set
if 'JAVA_HOME' not in os.environ:
    try:
        # Try to find java executable
        java_path = subprocess.check_output(['which', 'java'], text=True).strip()
        # Resolve symlinks to find actual JDK path
        if os.path.islink(java_path):
            java_path = os.path.realpath(java_path)
        java_home = os.path.dirname(os.path.dirname(java_path))
        os.environ['JAVA_HOME'] = java_home
        print(f"JAVA_HOME set to: {java_home}")
    except Exception as e:
        # Try common Nix store paths
        nix_java_paths = [
            '/nix/store/*-openjdk-*/lib/openjdk',
            '/nix/store/*-jdk-*/lib/openjdk',
        ]
        import glob
        for pattern in nix_java_paths:
            matches = glob.glob(pattern)
            if matches:
                os.environ['JAVA_HOME'] = matches[0]
                print(f"JAVA_HOME set to: {matches[0]}")
                break
        else:
            print(f"Warning: Could not auto-detect JAVA_HOME: {e}")

# Set SPARK_HOME to the installed Spark location
spark_home = '/home/runner/workspace/spark-3.5.0-bin-hadoop3'
if os.path.exists(spark_home):
    os.environ['SPARK_HOME'] = spark_home
    print(f"SPARK_HOME set to: {spark_home}")
else:
    print(f"Warning: Spark not found at {spark_home}")

import findspark
findspark.init(spark_home=spark_home if os.path.exists(spark_home) else None)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, rand, randn, abs as spark_abs, least, greatest, lit, udf, avg
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
import numpy as np
import pandas as pd
import os

class SparkCreditRiskGenerator:
    """Generate large-scale credit data with banking risk metrics using PySpark."""

    def __init__(self, n_samples=1000000):
        self.n_samples = n_samples
        self.spark = SparkSession.builder \
            .appName("ReverseSavingsCreditRisk") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

        # Set log level to reduce verbosity
        self.spark.sparkContext.setLogLevel("WARN")

    def generate_dataset(self):
        """Generate complete dataset with all features and credit risk metrics."""
        print(f"Generating {self.n_samples:,} samples using PySpark...")
        print("-" * 70)

        # Generate base features
        df = self.spark.range(self.n_samples)

        # Discipline features
        df = df.withColumn('on_time_rent_payments_pct',
                          greatest(lit(0), least(lit(100), 80 + randn() * 15)))
        df = df.withColumn('active_subscription_months',
                          greatest(lit(0), least(lit(60), 12 + spark_abs(randn()) * 15)))
        df = df.withColumn('utility_payment_consistency',
                          greatest(lit(0), least(lit(100), 85 + randn() * 10)))

        # Capacity features
        df = df.withColumn('monthly_net_income',
                          greatest(lit(1000), least(lit(8000), 3500 + randn() * 1500)))

        # Pay frequency
        df = df.withColumn('pay_freq_rand', rand())
        df = df.withColumn('pay_frequency',
                          when(col('pay_freq_rand') < 0.60, 'BIWEEKLY')
                          .when(col('pay_freq_rand') < 0.90, 'SEMIMONTHLY')
                          .otherwise('WEEKLY'))

        df = df.withColumn('paychecks_per_year',
                          when(col('pay_frequency') == 'WEEKLY', 52)
                          .when(col('pay_frequency') == 'BIWEEKLY', 26)
                          .otherwise(24))

        df = df.withColumn('paycheck_amount',
                          (col('monthly_net_income') * 12) / col('paychecks_per_year'))

        df = df.withColumn('monthly_rent',
                          col('monthly_net_income') * (0.25 + rand() * 0.15))
        df = df.withColumn('monthly_recurring_bills',
                          200 + rand() * 600)
        df = df.withColumn('alt_debt_to_income_ratio',
                          (col('monthly_rent') + col('monthly_recurring_bills')) / col('monthly_net_income'))

        # Stability features
        df = df.withColumn('income_variance',
                          col('monthly_net_income') * (0.05 + rand() * 0.25))
        df = df.withColumn('income_stability_index',
                          1 / greatest(lit(0.01), col('income_variance') / col('monthly_net_income')))
        df = df.withColumn('employment_tenure_months',
                          greatest(lit(0), least(lit(120), spark_abs(randn()) * 24)))
        df = df.withColumn('bank_account_age_months',
                          greatest(lit(6), least(lit(180), spark_abs(randn()) * 45)))

        # Composite score for risk calculation
        df = df.withColumn('discipline_score',
                          when((col('on_time_rent_payments_pct') > 85) &
                               (col('utility_payment_consistency') > 80) &
                               (col('active_subscription_months') > 6), 1).otherwise(0))
        df = df.withColumn('capacity_score',
                          when((col('alt_debt_to_income_ratio') < 0.6) &
                               (col('monthly_net_income') > 2000), 1).otherwise(0))
        df = df.withColumn('stability_score',
                          when((col('income_stability_index') > 3.5) &
                               (col('employment_tenure_months') > 3), 1).otherwise(0))

        df = df.withColumn('composite_score',
                          col('discipline_score') + col('capacity_score') + col('stability_score'))

        # === CREDIT RISK METRICS ===

        # 1. PD (Probability of Default) - ranges 0-1
        df = df.withColumn('noise_pd', rand())
        df = df.withColumn('PD',
                          when(col('composite_score') == 3, 0.05)
                          .when(col('composite_score') == 2, 0.30)
                          .when(col('composite_score') == 1, 0.70)
                          .otherwise(0.95))

        # 2. Repayment Status (realized default)
        df = df.withColumn('Repayment_Status',
                          when(col('noise_pd') < col('PD'), 1).otherwise(0))

        # 3. LGD (Loss Given Default) - 1 minus Recovery Rate
        # Typically ranges 0.4-0.7 for unsecured consumer loans
        df = df.withColumn('LGD',
                          greatest(lit(0.35), least(lit(0.85), 0.60 + randn() * 0.15)))

        # 4. RR (Recovery Rate) - complement of LGD
        df = df.withColumn('RR', 1 - col('LGD'))

        # 5. Max Safe Loan Amount (EAD basis)
        df = df.withColumn('three_month_income', col('monthly_net_income') * 3)
        df = df.withColumn('borrowing_capacity', col('three_month_income') * 0.15)
        df = df.withColumn('disposable_monthly',
                          greatest(lit(0), col('monthly_net_income') - col('monthly_rent') -
                                  col('monthly_recurring_bills') - 500))

        df = df.withColumn('Max_Safe_Loan_Amount',
                          when(col('disposable_monthly') < 200, 0)
                          .otherwise(greatest(lit(300), least(lit(3000),
                                              col('borrowing_capacity') * (0.5 + rand() * 0.5)))))

        # 6. EAD (Exposure at Default) - amount at risk
        # For this model, EAD = Max_Safe_Loan_Amount (assumed full drawdown)
        df = df.withColumn('EAD', col('Max_Safe_Loan_Amount'))

        # 7. ECL (Expected Credit Loss) = PD × LGD × EAD
        df = df.withColumn('ECL', col('PD') * col('LGD') * col('EAD'))

        # 8. ALLL (Allowance for Loan and Lease Losses) - reserve amount
        # ALLL is typically ECL + management buffer (10-20%)
        df = df.withColumn('ALLL', col('ECL') * (1.15 + rand() * 0.10))

        # Add subscription tier system
        df = df.withColumn("Base_Subscription_Fee", lit(16.12))

        # Premium payment delivery tiers
        # 15% Basic (free), 70% instant delivery options (split evenly), 15% family remittance
        premium_tier_udf = udf(lambda: int(np.random.choice(
            [0, 1, 2, 3, 4],  # 0=Basic, 1=Instant DD, 2=Cash Card, 3=Digital Wallet, 4=Family Remittance
            p=[0.15, 7/30, 7/30, 7/30, 0.15]  # Sums to 1.0: 15% basic, 70% instant (3 ways), 15% family
        )), IntegerType())

        df = df.withColumn("Premium_Tier_Code", premium_tier_udf())

        # Map tier codes to names and fees
        tier_mapping = {
            0: ("Basic_ACH", 0.00),
            1: ("Instant_DirectDeposit", 2.00),
            2: ("Physical_CashCard", 2.00),
            3: ("Digital_Wallet", 2.00),
            4: ("Family_Remittance", 2.00)
        }

        tier_name_udf = udf(lambda code: tier_mapping.get(code, ("Basic_ACH", 0.00))[0], StringType())
        tier_fee_udf = udf(lambda code: tier_mapping.get(code, ("Basic_ACH", 0.00))[1], DoubleType())

        df = df.withColumn("Premium_Tier_Name", tier_name_udf(col("Premium_Tier_Code")))
        df = df.withColumn("Premium_Add_On_Fee", tier_fee_udf(col("Premium_Tier_Code")))
        df = df.withColumn("Monthly_Subscription_Fee", col("Base_Subscription_Fee") + col("Premium_Add_On_Fee"))

        # Select final columns
        final_cols = [
            'on_time_rent_payments_pct',
            'active_subscription_months',
            'utility_payment_consistency',
            'monthly_net_income',
            'pay_frequency',
            'paycheck_amount',
            'paychecks_per_year',
            'monthly_rent',
            'monthly_recurring_bills',
            'alt_debt_to_income_ratio',
            'income_stability_index',
            'employment_tenure_months',
            'bank_account_age_months',
            'PD',
            'LGD',
            'RR',
            'EAD',
            'ECL',
            'ALLL',
            'Repayment_Status',
            'Max_Safe_Loan_Amount',
            'Base_Subscription_Fee',
            'Premium_Tier_Code',
            'Premium_Tier_Name',
            'Premium_Add_On_Fee',
            'Monthly_Subscription_Fee'
        ]

        df_final = df.select(final_cols)

        return df_final

    def save_to_csv(self, df, output_path='data/synthetic/synthetic_credit_data_1M.csv'):
        """Save PySpark DataFrame to CSV."""
        print(f"\nSaving to {output_path}...")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Coalesce to single partition for single CSV file
        df.coalesce(1).write.mode('overwrite').option('header', 'true').csv('data/synthetic/temp_spark_output')

        # Move the part file to final location
        import glob
        import shutil

        part_file = glob.glob('data/synthetic/temp_spark_output/part-*.csv')[0]
        shutil.move(part_file, output_path)
        shutil.rmtree('data/synthetic/temp_spark_output')

        # Verify the saved file
        import pandas as pd
        verification_df = pd.read_csv(output_path)
        actual_count = len(verification_df)
        
        print(f"✓ Saved to {output_path}")
        print(f"✓ Verified: {actual_count:,} samples written to file")

    def generate_summary_stats(self, df):
        """Generate summary statistics."""
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)

        # Default rate
        default_rate = df.select(col('Repayment_Status').cast('double')).agg({'Repayment_Status': 'avg'}).collect()[0][0]
        repayment_rate = 1 - default_rate # Corrected variable name
        print(f"\nDefault Rate: {default_rate:.2%}")
        print(f"Repayment Rate: {repayment_rate:.2%}")

        # Premium tier adoption summary
        tier_counts = df.groupBy("Premium_Tier_Name").count().orderBy("count", ascending=False).collect()
        print(f"\nPremium Tier Adoption:")
        for row in tier_counts:
            pct = (row['count'] / self.n_samples) * 100
            print(f"  {row['Premium_Tier_Name']}: {row['count']:,} ({pct:.1f}%)")

        avg_subscription = df.agg(avg("Monthly_Subscription_Fee")).collect()[0][0]
        print(f"\nAverage Subscription Fee: ${avg_subscription:.2f}/month")

        # ECL portfolio metrics
        total_ead = df.select(col('EAD')).agg({'EAD': 'sum'}).collect()[0][0]
        total_ecl = df.select(col('ECL')).agg({'ECL': 'sum'}).collect()[0][0]
        total_alll = df.select(col('ALLL')).agg({'ALLL': 'sum'}).collect()[0][0]

        print(f"\nPortfolio Credit Risk:")
        print(f"  Total EAD: ${total_ead:,.2f}")
        print(f"  Total ECL: ${total_ecl:,.2f}")
        print(f"  Total ALLL Reserve: ${total_alll:,.2f}")
        print(f"  ECL/EAD Ratio: {(total_ecl/total_ead)*100:.2f}%")


    def save_credit_risk_metrics_table(self, df, output_dir='data/summaries'):
        """Save comprehensive credit risk metrics table."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Get summary statistics
        stats_df = df.select(
            'PD',
            'LGD',
            'RR',
            'EAD',
            'ECL',
            'ALLL'
        ).summary('count', 'mean', 'stddev', 'min', 'max')

        stats_pd = stats_df.toPandas()

        # Calculate portfolio totals
        total_ead = df.select(col('EAD')).agg({'EAD': 'sum'}).collect()[0][0]
        total_ecl = df.select(col('ECL')).agg({'ECL': 'sum'}).collect()[0][0]
        total_alll = df.select(col('ALLL')).agg({'ALLL': 'sum'}).collect()[0][0]
        avg_pd = df.select(col('PD')).agg({'PD': 'avg'}).collect()[0][0]
        avg_lgd = df.select(col('LGD')).agg({'LGD': 'avg'}).collect()[0][0]
        avg_rr = df.select(col('RR')).agg({'RR': 'avg'}).collect()[0][0]

        # Save detailed statistics CSV
        stats_pd.to_csv(f'{output_dir}/credit_risk_metrics_detailed.csv', index=False)
        print(f"\n✓ Saved detailed metrics: {output_dir}/credit_risk_metrics_detailed.csv")

        # Create summary table
        summary_data = {
            'Metric': [
                'Total Exposure at Default (EAD)',
                'Total Expected Credit Loss (ECL)',
                'Total ALLL Reserve',
                'ECL/EAD Ratio',
                'Average Probability of Default (PD)',
                'Average Loss Given Default (LGD)',
                'Average Recovery Rate (RR)',
                'Number of Loans'
            ],
            'Value': [
                f'${total_ead:,.2f}',
                f'${total_ecl:,.2f}',
                f'${total_alll:,.2f}',
                f'{(total_ecl/total_ead)*100:.2f}%',
                f'{avg_pd:.4f}',
                f'{avg_lgd:.4f}',
                f'{avg_rr:.4f}',
                f'{df.count():,}'
            ],
            'Description': [
                'Total dollar amount at risk across all loans',
                'Expected loss from defaults (PD × LGD × EAD)',
                'Reserve amount set aside for loan losses',
                'Expected loss as percentage of total exposure',
                'Mean probability of borrower default',
                'Mean loss rate when default occurs',
                'Mean recovery rate from defaulted loans',
                'Total number of loan samples in dataset'
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/credit_risk_metrics_summary.csv', index=False)
        print(f"✓ Saved summary table: {output_dir}/credit_risk_metrics_summary.csv")

        # Create human-readable text report
        with open(f'{output_dir}/credit_risk_metrics_report.txt', 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CREDIT RISK METRICS SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("PORTFOLIO OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Loans: {df.count():,}\n")
            f.write(f"Total Exposure at Default (EAD): ${total_ead:,.2f}\n")
            f.write(f"Total Expected Credit Loss (ECL): ${total_ecl:,.2f}\n")
            f.write(f"Total ALLL Reserve: ${total_alll:,.2f}\n")
            f.write(f"ECL/EAD Ratio: {(total_ecl/total_ead)*100:.2f}%\n\n")

            f.write("AVERAGE RISK METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Probability of Default (PD): {avg_pd:.4f} ({avg_pd*100:.2f}%)\n")
            f.write(f"Loss Given Default (LGD): {avg_lgd:.4f} ({avg_lgd*100:.2f}%)\n")
            f.write(f"Recovery Rate (RR): {avg_rr:.4f} ({avg_rr*100:.2f}%)\n\n")

            f.write("METRIC DEFINITIONS\n")
            f.write("-" * 70 + "\n")
            f.write("PD: Probability that a borrower will default (0-1 scale)\n")
            f.write("LGD: Loss severity if default occurs (1 - Recovery Rate)\n")
            f.write("RR: Recovery Rate - portion of loan recovered after default\n")
            f.write("EAD: Exposure at Default - dollar amount at risk per loan\n")
            f.write("ECL: Expected Credit Loss = PD × LGD × EAD\n")
            f.write("ALLL: Allowance for Loan/Lease Losses - reserve amount\n")
            f.write("\n" + "=" * 70 + "\n")

        print(f"✓ Saved text report: {output_dir}/credit_risk_metrics_report.txt")

        return summary_df

    def stop(self):
        """Stop Spark session."""
        self.spark.stop()


def generate_spark_data(n_samples=5000000, output_path='data/synthetic/synthetic_credit_data_5000K.csv'):
    """Main function to generate large-scale dataset."""
    print("=" * 70)
    print(f"PYSPARK DATA GENERATION: {n_samples:,} SAMPLES")
    print("=" * 70)
    print()

    generator = SparkCreditRiskGenerator(n_samples=n_samples)

    # Generate dataset
    df = generator.generate_dataset()

    # Show summary statistics
    generator.generate_summary_stats(df)

    # Save credit risk metrics tables
    print("\nSaving Credit Risk Metrics Tables...")
    print("-" * 70)
    generator.save_credit_risk_metrics_table(df)

    # Save to CSV
    generator.save_to_csv(df, output_path)

    # Stop Spark
    generator.stop()

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE! ✓")
    print("=" * 70)
    print(f"\nCredit Risk Metrics Added:")
    print("  ✓ PD (Probability of Default)")
    print("  ✓ LGD (Loss Given Default)")
    print("  ✓ RR (Recovery Rate)")
    print("  ✓ EAD (Exposure at Default)")
    print("  ✓ ECL (Expected Credit Loss)")
    print("  ✓ ALLL (Allowance for Loan/Lease Losses)")
    print(f"\nTables Saved:")
    print("  ✓ data/summaries/credit_risk_metrics_summary.csv")
    print("  ✓ data/summaries/credit_risk_metrics_detailed.csv")
    print("  ✓ data/summaries/credit_risk_metrics_report.txt")
    print()


if __name__ == "__main__":
    generate_spark_data(n_samples=1000000)