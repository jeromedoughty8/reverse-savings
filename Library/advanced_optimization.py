
"""
Advanced Model Optimization
Combines hyperparameter tuning, feature engineering, and threshold optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
import xgboost as xgb
import joblib
import os

def create_interaction_features(X, feature_names):
    """Create interaction features for better predictions."""
    df = pd.DataFrame(X, columns=feature_names)
    
    # Key interactions based on domain knowledge
    df['income_stability_ratio'] = df['monthly_net_income'] / (df['income_stability_index'] + 1)
    df['payment_capacity'] = df['monthly_net_income'] * df['on_time_rent_payments_pct']
    df['employment_income'] = df['employment_tenure_months'] * df['monthly_net_income']
    df['debt_capacity'] = df['alt_debt_to_income_ratio'] * df['monthly_net_income']
    df['subscription_reliability'] = df['active_subscription_months'] * df['utility_payment_consistency']
    
    return df.values, list(df.columns)

def optimize_xgboost_hyperparameters(X_train, y_train):
    """Grid search for optimal XGBoost parameters."""
    print("Running hyperparameter optimization...")
    print("-" * 70)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [100, 150, 200],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'scale_pos_weight': [scale_pos_weight * 1.5, scale_pos_weight * 2.0]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )
    
    # Use F1 as primary metric
    f1_scorer = make_scorer(f1_score)
    
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=3,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    print(f"\n✓ Best F1-Score (CV): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def optimize_threshold(model, X_val, y_val):
    """Find optimal decision threshold to maximize F1 while maintaining recall."""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    results = []
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Prioritize F1 but ensure recall > 60%
        if recall >= 0.60 and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    results_df = pd.DataFrame(results)
    best_row = results_df[results_df['threshold'] == best_threshold].iloc[0]
    
    print(f"\n✓ Optimal threshold: {best_threshold:.3f}")
    print(f"  - Precision: {best_row['precision']:.2%}")
    print(f"  - Recall: {best_row['recall']:.2%}")
    print(f"  - F1-Score: {best_row['f1']:.2%}")
    
    return best_threshold, results_df

def run_advanced_optimization():
    """Run complete optimization pipeline."""
    print("=" * 70)
    print("ADVANCED MODEL OPTIMIZATION")
    print("=" * 70)
    print()
    
    # Load data
    data = np.load('data/models_output/processed_data.npz', allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_class_train']
    y_test = data['y_class_test']
    feature_names = data['feature_names']
    
    # Split test into validation and final test
    split_point = len(X_test) // 2
    X_val = X_test[:split_point]
    X_final_test = X_test[split_point:]
    y_val = y_test[:split_point]
    y_final_test = y_test[split_point:]
    
    print(f"Dataset sizes:")
    print(f"  - Train: {len(X_train):,}")
    print(f"  - Validation: {len(X_val):,}")
    print(f"  - Final Test: {len(X_final_test):,}")
    print()
    
    # Step 1: Feature Engineering
    print("Step 1: Creating interaction features...")
    print("-" * 70)
    X_train_eng, new_features = create_interaction_features(X_train, feature_names)
    X_val_eng, _ = create_interaction_features(X_val, feature_names)
    X_test_eng, _ = create_interaction_features(X_final_test, feature_names)
    print(f"✓ Original features: {len(feature_names)}")
    print(f"✓ New features added: {len(new_features) - len(feature_names)}")
    print(f"✓ Total features: {len(new_features)}")
    print()
    
    # Step 2: Hyperparameter Tuning
    print("Step 2: Hyperparameter optimization (this may take a few minutes)...")
    print("-" * 70)
    best_model = optimize_xgboost_hyperparameters(X_train_eng, y_train)
    print()
    
    # Step 3: Threshold Optimization
    print("Step 3: Threshold optimization...")
    print("-" * 70)
    optimal_threshold, threshold_results = optimize_threshold(best_model, X_val_eng, y_val)
    print()
    
    # Step 4: Evaluate on final test set
    print("Step 4: Final evaluation on test set...")
    print("-" * 70)
    y_pred_proba = best_model.predict_proba(X_test_eng)[:, 1]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    from Library.model_evaluation import ConfusionMatrixAnalyzer
    analyzer = ConfusionMatrixAnalyzer(y_final_test, y_pred, y_pred_proba)
    metrics = analyzer.print_business_interpretation()
    print()
    
    # Save optimized model
    os.makedirs('models/optimized', exist_ok=True)
    joblib.dump(best_model, 'models/optimized/xgboost_optimized.pkl')
    joblib.dump({
        'threshold': optimal_threshold,
        'features': new_features,
        'metrics': metrics
    }, 'models/optimized/optimization_config.pkl')
    
    # Save threshold analysis
    os.makedirs('data/optimization', exist_ok=True)
    threshold_results.to_csv('data/optimization/threshold_analysis.csv', index=False)
    
    print("=" * 70)
    print("OPTIMIZATION COMPLETE! ✓")
    print("=" * 70)
    print(f"✓ Optimized model saved to: models/optimized/xgboost_optimized.pkl")
    print(f"✓ Threshold analysis saved to: data/optimization/threshold_analysis.csv")
    print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
    print()
    
    return best_model, optimal_threshold, metrics

if __name__ == "__main__":
    run_advanced_optimization()
