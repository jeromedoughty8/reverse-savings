
# Reverse Savings Credit System

Alternative credit scoring system using a subscription-based liquidity model with advanced machine learning and custom analytical frameworks.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Dependencies](#dependencies)
- [Usage Guide](#usage-guide)
- [AI Tool Usage](#ai-tool-usage)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Technical Details](#technical-details)

---

## Overview

Reverse Savings provides 0% interest borrowing through a subscription-based platform ($16.12/month) designed for underbanked individuals. The system uses alternative credit data (rent payments, utility bills, subscription history) to assess creditworthiness without traditional credit scores.

**Key Innovation:** Custom cost analysis frameworks that optimize for business economics (FP cost = $100, FN cost = $1,000) rather than traditional ML metrics.

---

## Features

### Core Functionality
- **Subscription-based liquidity**: Pay-first model with 15% of 3-month income borrowing cap
- **0% interest borrowing**: No interest charges, only subscription fees
- **Alternative credit scoring**: Uses non-traditional data sources
- **XGBoost classification**: Primary model for default prediction

### Advanced Analytics
- **Threshold Stress Testing**: 20 threshold scenarios with cost-benefit analysis
- **Batched Cost Analysis**: FP/FN weighted analysis in 100-customer batches
- **Monte Carlo Simulation**: 10K iterations for portfolio risk assessment
- **SHAP Explainability**: Global and local model interpretations

### Dashboards
- **Streamlit Dashboard**: Interactive user interface with AI chatbot (Mistral-powered)
- **Data Upload**: CSV import for credit assessment
- **Analytics**: Real-time financial insights and predictions

---

## Setup Instructions

### 1. Environment Setup (Replit - Recommended)

This project is optimized for Replit and requires no additional setup:

1. Click the **Run** button to start the complete pipeline
2. Or use the **Dashboard** workflow to launch the Streamlit UI

### 2. Local Setup (Alternative)

If running locally:

```bash
# Clone repository
git clone <repository-url>
cd reverse-savings

# Install dependencies (uses uv package manager)
uv sync

# Run complete pipeline
python main.py all

# Or run dashboard
streamlit run dashboard/app.py --server.port=5000 --server.address=0.0.0.0
```

### 3. Configure API Keys (Optional - For Dashboard Chatbot)

To enable the Mistral AI chatbot in the dashboard:

1. Go to **Tools → Secrets** in Replit
2. Add a new secret:
   - Key: `MISTRAL_API_KEY`
   - Value: Your API key from [https://console.mistral.ai/](https://console.mistral.ai/)
3. Restart the Dashboard workflow

---

## Dependencies

### Core Dependencies (Managed by `pyproject.toml`)

```toml
python = ">=3.11"

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization
matplotlib>=3.10.7
seaborn>=0.13.2
plotly>=5.17.0

# Explainability
shap (via XGBoost installation)

# Dashboard
streamlit>=1.28.0
mistralai>=0.1.0

# Utilities
joblib>=1.3.0
python-docx>=1.2.0
markdown>=3.10
```

### System Requirements

- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for full 1M dataset)
- **Storage**: 2GB free space
- **OS**: Linux (Replit), macOS, or Windows

---

## Usage Guide

### Running the Complete Pipeline

#### Option 1: Interactive Mode (Recommended)
```bash
python main.py all
```

This will:
1. Prompt you for dataset size (10% to 100% of 1M samples)
2. Generate synthetic credit data
3. Preprocess and split data
4. Train baseline (Logistic Regression) and primary (XGBoost) models
5. Run regression models for loan amount prediction
6. Execute batched cost analysis
7. Generate SHAP explainability reports
8. Run Monte Carlo simulations
9. Perform threshold stress testing

**Expected Runtime:**
- 10% dataset (100K samples): ~2 minutes
- 50% dataset (500K samples): ~5-7 minutes
- 100% dataset (1M samples): ~10-12 minutes

#### Option 2: Individual Weeks
```bash
# Week 1: Data Generation
python main.py week1

# Week 2: Data Preparation & EDA
python main.py week2

# Week 3: Core Modeling
python main.py week3

# Week 4: Regression Modeling
python main.py regression

# Advanced Analytics
python main.py monte_carlo
python main.py stress_test
```

### Running the Dashboard

```bash
streamlit run dashboard/app.py --server.port=5000 --server.address=0.0.0.0
```

Or click the **Dashboard** workflow button in Replit.

**Dashboard Features:**
- 🏠 **Home**: Quick stats and recent activity
- 📊 **My Data**: Upload CSV files for credit assessment
- 💬 **Chat with Cash**: AI-powered financial assistant (requires Mistral API key)
- 💸 **Send Money**: Transfer simulation
- 📈 **Analytics**: Spending insights and predictions

---

## AI Tool Usage

AI tools, particularly **Replit Assistant**, helped with debugging, Streamlit dashboard setup, and documentation during development:

### Debugging Support

AI assistance was valuable for:
- Resolving WebSocket connection errors in Streamlit
- Fixing port binding issues (using `0.0.0.0` instead of `localhost`)
- Debugging confusion matrix label ordering
- Resolving SHAP compatibility issues with XGBoost parameters

### Streamlit Dashboard Setup

**Replit Assistant** helped with:
- Configuring Streamlit server settings for Replit environment
- Setting up Mistral API integration with proper error handling
- Managing environment variables through Replit Secrets
- Implementing the multi-tab dashboard structure
- Resolving state management issues in the chat interface

Example debugging scenarios:
- Diagnosing that Streamlit needed `--server.address=0.0.0.0` for Replit deployment
- Troubleshooting API authentication errors with Mistral
- Fixing file upload handling in the data management tab

### Documentation Assistance

AI tools also helped create this README document, including:
- Structuring the setup instructions and usage guide
- Formatting technical details and dependency lists
- Organizing project structure documentation

---

## Project Structure

```
reverse-savings/
├── Library/                        # Core ML modules
│   ├── data_synthesizer.py        # Synthetic data generation
│   ├── data_preparation.py        # Preprocessing & EDA
│   ├── core_modeling.py           # Classification models
│   ├── regression_modeling.py     # Loan amount prediction
│   ├── batched_cost_analysis.py   # FP/FN batch analysis
│   ├── threshold_stress_test.py   # Threshold optimization
│   ├── monte_carlo_simulation.py  # Portfolio risk simulation
│   ├── shap_explainability.py     # SHAP XAI implementation
│   └── model_evaluation.py        # Confusion matrix analysis
│
├── dashboard/                      # Streamlit UI
│   ├── app.py                     # Main dashboard application
│   └── README.md                  # Dashboard documentation
│
├── Report/                         # Documentation
│   ├── FINAL_TECHNICAL_REPORT.md  # Complete technical report
│   └── FINAL_TECHNICAL_REPORT.docx
│
├── data/                          # Generated outputs
│   ├── synthetic/                 # Synthetic datasets
│   ├── eda_plots/                 # Exploratory visualizations
│   ├── comparisons/               # Model comparisons
│   ├── batched_cost_analysis/     # Batch analysis results
│   ├── stress_test/               # Threshold testing
│   ├── monte_carlo/               # Risk simulations
│   └── SHAP/                      # Explainability reports
│
├── models/                        # Trained models
│   ├── logistic_regression_baseline.pkl
│   ├── xgboost_primary.pkl
│   └── regression/
│
├── main.py                        # Main execution script
├── convert_to_word.py             # Report converter
├── pyproject.toml                 # Dependencies
└── README.md                      # This file
```

---

## Output Files

### Generated Datasets
- `data/synthetic/synthetic_credit_data.csv` - Full synthetic dataset (configurable size)
- `data/models_output/processed_data.npz` - Preprocessed train/test splits

### Model Files
- `models/logistic_regression_baseline.pkl` - Baseline classifier
- `models/xgboost_primary.pkl` - Primary XGBoost classifier
- `models/regression/xgboost_regression.pkl` - Loan amount predictor

### Visualizations
- `data/comparisons/model_comparison.png` - Classification model comparison
- `data/batched_cost_analysis/batched_cost_analysis.png` - Batch-level FP/FN costs
- `data/stress_test/threshold_heatmap_comparison.png` - Threshold performance heatmap
- `data/monte_carlo/monte_carlo_simulation.png` - Portfolio risk distributions
- `data/SHAP/Classification/global_summary_plot.png` - SHAP feature importance

### Reports
- `data/summaries/ml_outcomes_summary.txt` - ML metrics summary
- `data/monte_carlo/monte_carlo_report.txt` - Risk analysis report
- `Report/FINAL_TECHNICAL_REPORT.docx` - Complete technical documentation

---

## Technical Details

### Data Generation
- **Size**: Configurable (100K to 1M samples)
- **Features**: 20 alternative credit features (discipline, capacity, stability)
- **Targets**: Binary classification (default/repay) + regression (loan amount)
- **Default Rate**: ~15% (matches federal student loan benchmark)

### Machine Learning Models

#### Classification
- **Baseline**: Logistic Regression (L2 regularization)
- **Primary**: XGBoost (100 estimators, max_depth=6)
- **Optimization**: Custom threshold tuning (default: 0.35 for higher recall)

#### Regression
- **Models**: Linear, Ridge, XGBoost
- **Target**: Max safe loan amount ($300-$3,000 range)
- **Metric**: MAE (Mean Absolute Error) and custom MAAE (Mean Asymmetric Absolute Error)

### Custom Analytical Frameworks

#### 1. Batched Cost Analysis
- **Batch Size**: 100 customers
- **Cost Formula**: FP + FN×10
- **Threshold**: Batches with cost ≥ 2
- **Opportunity Loss**: (Sum - Frequency) × $1,000

#### 2. Threshold Stress Test
- **Thresholds**: 5% to 100% in 5% increments (20 scenarios)
- **Metrics**: Net profit, ECL, approval rate, cost ratio
- **Cost Ratio**: (FP × 0.1) / FN (break-even at 1.0)

#### 3. Monte Carlo Simulation
- **Iterations**: 10,000 runs
- **Variance Reduction**: Antithetic variates enabled
- **Outputs**: VaR (95%), CVaR, expected losses, revenue projections

### Explainability (SHAP)
- **Method**: TreeExplainer (optimized for XGBoost)
- **Global**: Feature importance rankings
- **Local**: 4 case studies (TP, TN, FP, FN)
- **Bias Check**: Validates ethical feature usage

---

## Support & Documentation

- **Technical Report**: See `Report/FINAL_TECHNICAL_REPORT.md`
- **Dashboard Guide**: See `dashboard/README.md`
- **API Integration**: See `Report/API_INTEGRATION_GUIDE.md`
- **Subscription Model**: See `Report/SUBSCRIPTION_MODEL.md`

---

## Credits

**Project**: Reverse Savings Credit System  
**Framework**: Alternative credit scoring with ML  
**AI Tools Used**: Replit Assistant, ChatGPT  
**Key Technologies**: Python, XGBoost, SHAP, Streamlit, Mistral AI

---

## License

This project is for educational and research purposes.
