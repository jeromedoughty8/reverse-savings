
"""
Monte Carlo Simulation for Reverse Savings
Risk Analysis, Revenue Projections, and Business Viability Testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class MonteCarloCreditRisk:
    """Monte Carlo simulation for credit risk and business metrics."""
    
    def __init__(self, n_simulations=10000, random_seed=42, use_antithetic=True):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulation runs
            random_seed: Random seed for reproducibility
            use_antithetic: Whether to use antithetic variance reduction
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.use_antithetic = use_antithetic
        np.random.seed(random_seed)
        
    def simulate_default_rates(self, base_default_rate=0.15, volatility=0.05):
        """
        Simulate distribution of possible default rates.
        
        Args:
            base_default_rate: Expected default rate (e.g., 15%)
            volatility: Standard deviation of default rate
            
        Returns:
            Array of simulated default rates
        """
        # Use beta distribution (bounded between 0 and 1)
        alpha, beta = self._fit_beta_params(base_default_rate, volatility)
        default_rates = np.random.beta(alpha, beta, self.n_simulations)
        
        return default_rates
    
    def simulate_portfolio_performance(self, n_borrowers=1000, avg_loan=700, 
                                      default_rate=0.15, lgd=0.40):
        """
        Simulate portfolio-level default losses with antithetic variance reduction.
        
        Args:
            n_borrowers: Number of active borrowers
            avg_loan: Average loan amount
            default_rate: Expected default rate
            lgd: Loss Given Default (% of loan lost on default)
            
        Returns:
            DataFrame with simulation results
        """
        results = []
        
        if self.use_antithetic:
            # Use antithetic variance reduction (pairs of simulations)
            half_sims = self.n_simulations // 2
            
            for i in range(half_sims):
                # Generate standard uniform random numbers
                u_defaults = np.random.uniform(0, 1, n_borrowers)
                u_loans = np.random.normal(0, 1, n_borrowers)
                
                # STANDARD SIMULATION
                defaults = (u_defaults < default_rate).astype(int)
                n_defaults = defaults.sum()
                loan_amounts = np.exp(np.log(avg_loan) + 0.3 * u_loans)
                
                defaulted_loans = loan_amounts[defaults == 1]
                total_exposure = loan_amounts.sum()
                total_loss = defaulted_loans.sum() * lgd
                loss_rate = total_loss / total_exposure if total_exposure > 0 else 0
                
                results.append({
                    'simulation': i * 2,
                    'n_defaults': n_defaults,
                    'default_rate': n_defaults / n_borrowers,
                    'total_exposure': total_exposure,
                    'total_loss': total_loss,
                    'loss_rate': loss_rate
                })
                
                # ANTITHETIC SIMULATION (using complementary random numbers)
                u_defaults_anti = 1 - u_defaults
                u_loans_anti = -u_loans
                
                defaults_anti = (u_defaults_anti < default_rate).astype(int)
                n_defaults_anti = defaults_anti.sum()
                loan_amounts_anti = np.exp(np.log(avg_loan) + 0.3 * u_loans_anti)
                
                defaulted_loans_anti = loan_amounts_anti[defaults_anti == 1]
                total_exposure_anti = loan_amounts_anti.sum()
                total_loss_anti = defaulted_loans_anti.sum() * lgd
                loss_rate_anti = total_loss_anti / total_exposure_anti if total_exposure_anti > 0 else 0
                
                results.append({
                    'simulation': i * 2 + 1,
                    'n_defaults': n_defaults_anti,
                    'default_rate': n_defaults_anti / n_borrowers,
                    'total_exposure': total_exposure_anti,
                    'total_loss': total_loss_anti,
                    'loss_rate': loss_rate_anti
                })
        else:
            # Standard Monte Carlo without variance reduction
            for i in range(self.n_simulations):
                defaults = np.random.binomial(1, default_rate, n_borrowers)
                n_defaults = defaults.sum()
                loan_amounts = np.random.lognormal(np.log(avg_loan), 0.3, n_borrowers)
                
                defaulted_loans = loan_amounts[defaults == 1]
                total_exposure = loan_amounts.sum()
                total_loss = defaulted_loans.sum() * lgd
                loss_rate = total_loss / total_exposure if total_exposure > 0 else 0
                
                results.append({
                    'simulation': i,
                    'n_defaults': n_defaults,
                    'default_rate': n_defaults / n_borrowers,
                    'total_exposure': total_exposure,
                    'total_loss': total_loss,
                    'loss_rate': loss_rate
                })
        
        return pd.DataFrame(results)
    
    def simulate_revenue_projections(self, base_subscribers=1000, 
                                    monthly_subscription=16.12,
                                    churn_rate=0.05, growth_rate=0.10,
                                    months=12):
        """
        Simulate monthly revenue under uncertainty with antithetic variance reduction.
        
        Args:
            base_subscribers: Starting subscriber count
            monthly_subscription: Subscription fee
            churn_rate: Monthly churn rate (5% = 0.05)
            growth_rate: Monthly growth rate (10% = 0.10)
            months: Projection period in months
            
        Returns:
            DataFrame with revenue projections
        """
        results = []
        
        if self.use_antithetic:
            half_sims = self.n_simulations // 2
            
            for i in range(half_sims):
                # Generate random numbers for standard simulation
                u_growth = np.random.normal(0, 1, months)
                
                # STANDARD SIMULATION
                subscribers = base_subscribers
                monthly_revenues = []
                
                for month in range(months):
                    churned = np.random.binomial(subscribers, churn_rate)
                    growth_volatility = growth_rate * 0.3
                    actual_growth = max(0, growth_rate + growth_volatility * u_growth[month])
                    new_subscribers = int(subscribers * actual_growth)
                    subscribers = subscribers - churned + new_subscribers
                    monthly_revenues.append(subscribers * monthly_subscription)
                
                results.append({
                    'simulation': i * 2,
                    'final_subscribers': subscribers,
                    'total_revenue_12mo': sum(monthly_revenues),
                    'avg_monthly_revenue': np.mean(monthly_revenues),
                    'revenue_volatility': np.std(monthly_revenues)
                })
                
                # ANTITHETIC SIMULATION
                u_growth_anti = -u_growth
                subscribers_anti = base_subscribers
                monthly_revenues_anti = []
                
                for month in range(months):
                    churned_anti = np.random.binomial(subscribers_anti, churn_rate)
                    growth_volatility = growth_rate * 0.3
                    actual_growth_anti = max(0, growth_rate + growth_volatility * u_growth_anti[month])
                    new_subscribers_anti = int(subscribers_anti * actual_growth_anti)
                    subscribers_anti = subscribers_anti - churned_anti + new_subscribers_anti
                    monthly_revenues_anti.append(subscribers_anti * monthly_subscription)
                
                results.append({
                    'simulation': i * 2 + 1,
                    'final_subscribers': subscribers_anti,
                    'total_revenue_12mo': sum(monthly_revenues_anti),
                    'avg_monthly_revenue': np.mean(monthly_revenues_anti),
                    'revenue_volatility': np.std(monthly_revenues_anti)
                })
        else:
            # Standard Monte Carlo
            for i in range(self.n_simulations):
                subscribers = base_subscribers
                monthly_revenues = []
                
                for month in range(months):
                    churned = np.random.binomial(subscribers, churn_rate)
                    growth_volatility = growth_rate * 0.3
                    actual_growth = max(0, np.random.normal(growth_rate, growth_volatility))
                    new_subscribers = int(subscribers * actual_growth)
                    subscribers = subscribers - churned + new_subscribers
                    monthly_revenues.append(subscribers * monthly_subscription)
                
                results.append({
                    'simulation': i,
                    'final_subscribers': subscribers,
                    'total_revenue_12mo': sum(monthly_revenues),
                    'avg_monthly_revenue': np.mean(monthly_revenues),
                    'revenue_volatility': np.std(monthly_revenues)
                })
        
        return pd.DataFrame(results)
    
    def calculate_var_and_cvar(self, losses, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall).
        
        Args:
            losses: Array of simulated losses
            confidence_level: Confidence level (e.g., 95%)
            
        Returns:
            Dictionary with VaR and CVaR
        """
        var = np.percentile(losses, confidence_level * 100)
        cvar = losses[losses >= var].mean()
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level
        }
    
    def stress_test_scenarios(self, base_params):
        """
        Run stress tests under extreme scenarios.
        
        Args:
            base_params: Dictionary with base case parameters
            
        Returns:
            DataFrame with stress test results
        """
        scenarios = {
            'Base Case': {
                'default_rate': 0.15,
                'lgd': 0.40,
                'churn_rate': 0.05,
                'growth_rate': 0.10
            },
            'Mild Recession': {
                'default_rate': 0.25,
                'lgd': 0.50,
                'churn_rate': 0.08,
                'growth_rate': 0.05
            },
            'Severe Recession': {
                'default_rate': 0.40,
                'lgd': 0.60,
                'churn_rate': 0.15,
                'growth_rate': -0.05
            },
            'Best Case': {
                'default_rate': 0.08,
                'lgd': 0.30,
                'churn_rate': 0.03,
                'growth_rate': 0.20
            }
        }
        
        results = []
        
        for scenario_name, params in scenarios.items():
            # Portfolio simulation
            portfolio_df = self.simulate_portfolio_performance(
                n_borrowers=base_params.get('n_borrowers', 1000),
                avg_loan=base_params.get('avg_loan', 700),
                default_rate=params['default_rate'],
                lgd=params['lgd']
            )
            
            # Revenue simulation
            revenue_df = self.simulate_revenue_projections(
                base_subscribers=base_params.get('base_subscribers', 1000),
                churn_rate=params['churn_rate'],
                growth_rate=params['growth_rate']
            )
            
            results.append({
                'scenario': scenario_name,
                'avg_default_rate': portfolio_df['default_rate'].mean(),
                'avg_loss_rate': portfolio_df['loss_rate'].mean(),
                'var_95': self.calculate_var_and_cvar(portfolio_df['total_loss'])['var'],
                'cvar_95': self.calculate_var_and_cvar(portfolio_df['total_loss'])['cvar'],
                'avg_12mo_revenue': revenue_df['total_revenue_12mo'].mean(),
                'final_subscribers': revenue_df['final_subscribers'].mean()
            })
        
        return pd.DataFrame(results)
    
    def _fit_beta_params(self, mean, std):
        """Fit beta distribution parameters from mean and std."""
        if mean <= 0 or mean >= 1:
            raise ValueError("Mean must be between 0 and 1")
        
        variance = std ** 2
        alpha = mean * ((mean * (1 - mean) / variance) - 1)
        beta = (1 - mean) * ((mean * (1 - mean) / variance) - 1)
        
        return max(alpha, 0.1), max(beta, 0.1)
    
    def plot_simulation_results(self, portfolio_df, revenue_df, output_dir='data/monte_carlo'):
        """Create comprehensive visualization of simulation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Monte Carlo Simulation Results ({self.n_simulations:,} runs)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Default Rate Distribution
        ax1 = axes[0, 0]
        ax1.hist(portfolio_df['default_rate'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(portfolio_df['default_rate'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {portfolio_df['default_rate'].mean():.2%}")
        ax1.axvline(0.15, color='green', linestyle='--', label='Federal Benchmark: 15%')
        ax1.set_xlabel('Default Rate')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Default Rates')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Total Loss Distribution
        ax2 = axes[0, 1]
        losses = portfolio_df['total_loss']
        ax2.hist(losses, bins=50, color='coral', alpha=0.7, edgecolor='black')
        var_95 = self.calculate_var_and_cvar(losses)
        ax2.axvline(var_95['var'], color='red', linestyle='--', 
                   label=f"VaR (95%): ${var_95['var']:,.0f}")
        ax2.axvline(var_95['cvar'], color='darkred', linestyle='--', 
                   label=f"CVaR (95%): ${var_95['cvar']:,.0f}")
        ax2.set_xlabel('Total Loss ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Portfolio Loss Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Loss Rate Distribution
        ax3 = axes[0, 2]
        ax3.hist(portfolio_df['loss_rate'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax3.axvline(portfolio_df['loss_rate'].mean(), color='red', linestyle='--',
                   label=f"Mean: {portfolio_df['loss_rate'].mean():.2%}")
        ax3.set_xlabel('Loss Rate (% of Exposure)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Rate Distribution')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Revenue Distribution
        ax4 = axes[1, 0]
        ax4.hist(revenue_df['total_revenue_12mo'], bins=50, color='green', alpha=0.7, edgecolor='black')
        ax4.axvline(revenue_df['total_revenue_12mo'].mean(), color='red', linestyle='--',
                   label=f"Mean: ${revenue_df['total_revenue_12mo'].mean():,.0f}")
        ax4.set_xlabel('12-Month Revenue ($)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Annual Revenue Distribution')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Final Subscribers Distribution
        ax5 = axes[1, 1]
        ax5.hist(revenue_df['final_subscribers'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax5.axvline(revenue_df['final_subscribers'].mean(), color='red', linestyle='--',
                   label=f"Mean: {revenue_df['final_subscribers'].mean():,.0f}")
        ax5.set_xlabel('Final Subscriber Count')
        ax5.set_ylabel('Frequency')
        ax5.set_title('12-Month Subscriber Growth')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Risk-Return Scatter
        ax6 = axes[1, 2]
        ax6.scatter(portfolio_df['loss_rate'], revenue_df['total_revenue_12mo'], 
                   alpha=0.3, s=10, color='navy')
        ax6.set_xlabel('Loss Rate (%)')
        ax6.set_ylabel('12-Month Revenue ($)')
        ax6.set_title('Risk vs. Revenue Trade-off')
        ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/monte_carlo_simulation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Simulation plot saved: {output_dir}/monte_carlo_simulation.png")
    
    def generate_summary_report(self, portfolio_df, revenue_df, stress_df, output_dir='data/monte_carlo'):
        """Generate comprehensive text report."""
        os.makedirs(output_dir, exist_ok=True)
        
        var_results = self.calculate_var_and_cvar(portfolio_df['total_loss'])
        
        report = f"""
========================================================================
MONTE CARLO SIMULATION REPORT
Reverse Savings Credit Risk & Business Viability Analysis
========================================================================
Simulation Parameters: {self.n_simulations:,} runs

PORTFOLIO RISK METRICS:
------------------------------------------------------------------------
Default Rate:
  • Mean:              {portfolio_df['default_rate'].mean():.2%}
  • Std Dev:           {portfolio_df['default_rate'].std():.2%}
  • 5th Percentile:    {portfolio_df['default_rate'].quantile(0.05):.2%}
  • 95th Percentile:   {portfolio_df['default_rate'].quantile(0.95):.2%}
  • Federal Benchmark: 15.00%

Expected Losses:
  • Mean Loss:         ${portfolio_df['total_loss'].mean():,.2f}
  • Std Dev:           ${portfolio_df['total_loss'].std():,.2f}
  • VaR (95%):         ${var_results['var']:,.2f}
  • CVaR (95%):        ${var_results['cvar']:,.2f}

Loss Rate (% of Exposure):
  • Mean:              {portfolio_df['loss_rate'].mean():.2%}
  • 95th Percentile:   {portfolio_df['loss_rate'].quantile(0.95):.2%}

REVENUE PROJECTIONS (12-Month):
------------------------------------------------------------------------
Total Revenue:
  • Mean:              ${revenue_df['total_revenue_12mo'].mean():,.2f}
  • Std Dev:           ${revenue_df['total_revenue_12mo'].std():,.2f}
  • 5th Percentile:    ${revenue_df['total_revenue_12mo'].quantile(0.05):,.2f}
  • 95th Percentile:   ${revenue_df['total_revenue_12mo'].quantile(0.95):,.2f}

Subscriber Growth:
  • Mean Final Count:  {revenue_df['final_subscribers'].mean():,.0f}
  • Growth Range:      {revenue_df['final_subscribers'].quantile(0.05):,.0f} - {revenue_df['final_subscribers'].quantile(0.95):,.0f}

STRESS TEST SCENARIOS:
------------------------------------------------------------------------
{stress_df.to_string(index=False)}

RISK ASSESSMENT:
------------------------------------------------------------------------
• Probability of default rate > 15%: {(portfolio_df['default_rate'] > 0.15).mean():.1%}
• Probability of default rate > 20%: {(portfolio_df['default_rate'] > 0.20).mean():.1%}
• Probability of losses > $100K:     {(portfolio_df['total_loss'] > 100000).mean():.1%}

BUSINESS VIABILITY:
------------------------------------------------------------------------
Revenue Confidence Intervals (95%):
  Lower Bound: ${revenue_df['total_revenue_12mo'].quantile(0.025):,.2f}
  Upper Bound: ${revenue_df['total_revenue_12mo'].quantile(0.975):,.2f}

Risk-Adjusted Return:
  Expected Revenue:  ${revenue_df['total_revenue_12mo'].mean():,.2f}
  Expected Loss:     ${portfolio_df['total_loss'].mean():,.2f}
  Net Profit:        ${revenue_df['total_revenue_12mo'].mean() - portfolio_df['total_loss'].mean():,.2f}
  Profit Margin:     {((revenue_df['total_revenue_12mo'].mean() - portfolio_df['total_loss'].mean()) / revenue_df['total_revenue_12mo'].mean()):.1%}

========================================================================
RECOMMENDATIONS:
========================================================================
1. Capital Reserves: Maintain at least ${var_results['cvar']:,.0f} (CVaR 95%)
2. Default Rate Target: Keep below 15% (currently {portfolio_df['default_rate'].mean():.1%})
3. Revenue Diversification: Focus on subscriber growth to offset credit losses
4. Stress Testing: Prepare for scenarios with {stress_df.loc[stress_df['scenario']=='Severe Recession', 'avg_default_rate'].values[0]:.1%} default rates

========================================================================
"""
        
        with open(f'{output_dir}/monte_carlo_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print(f"✓ Report saved: {output_dir}/monte_carlo_report.txt")


def run_monte_carlo_analysis():
    """Main function to run Monte Carlo analysis."""
    print("=" * 70)
    print("MONTE CARLO SIMULATION - CREDIT RISK & BUSINESS ANALYSIS")
    print("=" * 70)
    print()
    
    # Initialize simulator with antithetic variance reduction
    mc = MonteCarloCreditRisk(n_simulations=10000, use_antithetic=True)
    print(f"Variance Reduction: Antithetic Variates {'ENABLED' if mc.use_antithetic else 'DISABLED'}")
    print(f"  → Reduces variance without extra computational cost")
    print(f"  → Improves convergence and confidence intervals")
    print()
    
    # Base parameters (1,000 subscribers case)
    base_params = {
        'n_borrowers': 400,  # 40% of 1,000 actively borrow
        'avg_loan': 700,
        'base_subscribers': 1000
    }
    
    print("Running portfolio performance simulation...")
    portfolio_df = mc.simulate_portfolio_performance(
        n_borrowers=base_params['n_borrowers'],
        avg_loan=base_params['avg_loan'],
        default_rate=0.15,
        lgd=0.40
    )
    print(f"✓ Completed {len(portfolio_df):,} portfolio simulations")
    print()
    
    print("Running revenue projections simulation...")
    revenue_df = mc.simulate_revenue_projections(
        base_subscribers=base_params['base_subscribers'],
        monthly_subscription=16.12,
        churn_rate=0.05,
        growth_rate=0.10,
        months=12
    )
    print(f"✓ Completed {len(revenue_df):,} revenue simulations")
    print()
    
    print("Running stress test scenarios...")
    stress_df = mc.stress_test_scenarios(base_params)
    print(f"✓ Completed stress tests for {len(stress_df)} scenarios")
    print()
    
    # Save results
    output_dir = 'data/monte_carlo'
    os.makedirs(output_dir, exist_ok=True)
    
    portfolio_df.to_csv(f'{output_dir}/portfolio_simulations.csv', index=False)
    revenue_df.to_csv(f'{output_dir}/revenue_simulations.csv', index=False)
    stress_df.to_csv(f'{output_dir}/stress_test_results.csv', index=False)
    
    print(f"✓ Simulation data saved to: {output_dir}/")
    print()
    
    # Generate visualizations
    print("Creating visualizations...")
    mc.plot_simulation_results(portfolio_df, revenue_df, output_dir)
    print()
    
    # Generate report
    print("Generating summary report...")
    mc.generate_summary_report(portfolio_df, revenue_df, stress_df, output_dir)
    
    print()
    print("=" * 70)
    print("MONTE CARLO ANALYSIS COMPLETE! ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_monte_carlo_analysis()
