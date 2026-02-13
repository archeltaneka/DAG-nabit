"""
Double Machine Learning (DML)
==============================

Modern causal inference using machine learning.

The idea:
1. Use ML to predict BOTH treatment and outcome
2. Use residuals to remove confounding
3. Estimate treatment effect on "de-confounded" data
4. Allows for flexible, non-linear relationships!

Key advantage: Can use powerful ML models (XGBoost, Random Forest)
while still getting valid causal estimates.

Reference: Chernozhukov et al. (2018)
Uses: Microsoft's EconML library
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats

# Try to import EconML - it may not be installed
try:
    from econml.dml import LinearDML, CausalForestDML
    from econml.inference import BootstrapInference
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    print("⚠️  EconML not installed. Install with: pip install econml")


class DoubleMachineLearning:
    """
    Double Machine Learning for causal inference.
    
    Uses cross-fitting and ML models to estimate treatment effects
    while controlling for confounders.
    
    More flexible than propensity score matching:
    - Can capture non-linear relationships
    - More robust to model misspecification
    - Can estimate heterogeneous effects
    """
    
    def __init__(
        self,
        treatment_col: str = 'treated',
        outcome_col: str = 'purchased',
        confounders: List[str] = None,
        model_type: str = 'linear',
        n_splits: int = 2
    ):
        """
        Args:
            treatment_col: Name of treatment variable
            outcome_col: Name of outcome variable
            confounders: List of confounder variables
            model_type: 'linear' or 'forest' (causal forest)
            n_splits: Number of cross-fitting folds
        """
        if not ECONML_AVAILABLE:
            raise ImportError("EconML is required. Install with: pip install econml")
        
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounders = confounders or []
        self.model_type = model_type
        self.n_splits = n_splits
        
        self.model = None
        self.cate_estimates = None
    
    def fit(self, data: pd.DataFrame) -> 'DoubleMachineLearning':
        """
        Fit the DML model.
        
        Returns:
            self (for method chaining)
        """
        X = data[self.confounders].values
        T = data[self.treatment_col].values
        Y = data[self.outcome_col].values
        
        if self.model_type == 'linear':
            # Linear DML - assumes constant treatment effect
            # But uses ML for nuisance functions
            self.model = LinearDML(
                model_y='auto',  # Auto-select Y model
                model_t='auto',  # Auto-select T model
                discrete_treatment=True,
                cv=self.n_splits,
                random_state=42
            )
        
        elif self.model_type == 'forest':
            # Causal Forest - estimates heterogeneous effects
            self.model = CausalForestDML(
                model_y='auto',
                model_t='auto',
                discrete_treatment=True,
                cv=self.n_splits,
                random_state=42,
                n_estimators=100
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Fit the model
        self.model.fit(Y, T, X=X)
        
        return self
    
    def estimate_ate(self, data: pd.DataFrame) -> Dict:
        """
        Estimate Average Treatment Effect (ATE).
        
        Args:
            data: DataFrame with features (must be provided)
        
        Returns:
            Dictionary with ATE estimate and confidence intervals
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        # Get features
        X = data[self.confounders].values
        
        # Get ATE (average over all observations)
        ate = self.model.ate(X=X)
        
        # Get confidence interval
        # Different EconML versions have different APIs, so try multiple approaches
        try:
            # Approach 1: Direct inference
            ate_inference = self.model.ate_inference(X=X)
            
            # Try to get CI from summary_frame first (most robust)
            if hasattr(ate_inference, 'summary_frame'):
                summary = ate_inference.summary_frame()
                ci_lower = summary.iloc[0]['mean_lower']
                ci_upper = summary.iloc[0]['mean_upper']
            # Try conf_int_mean method
            elif hasattr(ate_inference, 'conf_int_mean'):
                ci = ate_inference.conf_int_mean(alpha=0.05)
                ci_lower = ci[0]
                ci_upper = ci[1]
            # Try direct conf_int
            elif hasattr(ate_inference, 'conf_int'):
                ci = ate_inference.conf_int(alpha=0.05)
                ci_lower = ci[0]
                ci_upper = ci[1]
            else:
                raise AttributeError("Could not find CI method")
                
        except Exception as e:
            # Fallback: use bootstrap
            print(f"⚠️  Using bootstrap for CI (EconML API variation)")
            
            # Bootstrap confidence interval
            n_bootstrap = 100
            bootstrap_ates = []
            for _ in range(n_bootstrap):
                boot_idx = np.random.choice(len(X), len(X), replace=True)
                boot_ate = self.model.ate(X=X[boot_idx])
                bootstrap_ates.append(boot_ate)
            
            ci_lower = np.percentile(bootstrap_ates, 2.5)
            ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        return {
            'ate': float(ate),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'method': 'Double Machine Learning',
            'model_type': self.model_type
        }
    
    def estimate_cate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Estimate Conditional Average Treatment Effects (CATE).
        
        Returns treatment effect for each individual based on their features.
        This is what makes DML powerful - heterogeneous effects!
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        X = data[self.confounders].values
        self.cate_estimates = self.model.effect(X)
        
        return self.cate_estimates
    
    def get_top_responders(
        self,
        data: pd.DataFrame,
        top_k: int = 100
    ) -> pd.DataFrame:
        """
        Identify customers most likely to respond to treatment.
        
        This is the key business value: targeting!
        """
        if self.cate_estimates is None:
            self.estimate_cate(data)
        
        data_copy = data.copy()
        data_copy['predicted_effect'] = self.cate_estimates
        
        # Sort by predicted effect
        top_responders = data_copy.nlargest(top_k, 'predicted_effect')
        
        return top_responders
    
    def feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance for heterogeneity (Causal Forest only).
        
        Shows which features create the most variation in treatment effects.
        """
        if self.model_type != 'forest':
            return None
        
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance = pd.DataFrame({
            'feature': self.confounders,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def report(self, results: Dict) -> str:
        """Generate formatted report"""
        report = []
        report.append("="*70)
        report.append("DOUBLE MACHINE LEARNING RESULTS")
        report.append("="*70)
        
        report.append(f"\nMethod: {results['method']}")
        report.append(f"Model Type: {results['model_type']}")
        
        report.append(f"\nAverage Treatment Effect (ATE):")
        report.append(f"  Estimate: {results['ate']:+.4f}")
        report.append(f"  95% CI: [{results['ci_lower']:+.4f}, {results['ci_upper']:+.4f}]")
        
        is_significant = not (results['ci_lower'] < 0 < results['ci_upper'])
        report.append(f"  Significant: {'YES ✅' if is_significant else 'NO ❌'}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


class SimpleDoubleMachineLearning:
    """
    Simplified DML implementation without EconML dependency.
    
    Uses sklearn only - good for understanding the algorithm.
    Less feature-rich than EconML version.
    """
    
    def __init__(
        self,
        treatment_col: str = 'treated',
        outcome_col: str = 'purchased',
        confounders: List[str] = None
    ):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounders = confounders or []
        
        # Models for nuisance functions
        self.y_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.t_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.ate = None
    
    def fit_and_estimate(self, data: pd.DataFrame) -> Dict:
        """
        Fit DML and estimate ATE.
        
        Algorithm:
        1. Predict Y from X (outcome model)
        2. Predict T from X (treatment model)
        3. Get residuals
        4. Regress Y_residual on T_residual
        """
        X = data[self.confounders].values
        T = data[self.treatment_col].values
        Y = data[self.outcome_col].values
        
        # Step 1: Predict outcome from confounders
        self.y_model.fit(X, Y)
        y_pred = self.y_model.predict(X)
        y_residual = Y - y_pred
        
        # Step 2: Predict treatment from confounders
        self.t_model.fit(X, T)
        t_pred = self.t_model.predict_proba(X)[:, 1]
        t_residual = T - t_pred
        
        # Step 3: Regress residuals (this is the causal effect!)
        # ATE = Cov(Y_residual, T_residual) / Var(T_residual)
        self.ate = np.cov(y_residual, t_residual)[0, 1] / np.var(t_residual)
        
        # Bootstrap for confidence interval
        n_bootstrap = 1000
        bootstrap_ates = []
        
        for _ in range(n_bootstrap):
            # Resample
            indices = np.random.choice(len(data), size=len(data), replace=True)
            
            X_boot = X[indices]
            T_boot = T[indices]
            Y_boot = Y[indices]
            
            # Refit and estimate
            y_pred_boot = self.y_model.predict(X_boot)
            t_pred_boot = self.t_model.predict_proba(X_boot)[:, 1]
            
            y_res_boot = Y_boot - y_pred_boot
            t_res_boot = T_boot - t_pred_boot
            
            ate_boot = np.cov(y_res_boot, t_res_boot)[0, 1] / np.var(t_res_boot)
            bootstrap_ates.append(ate_boot)
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        return {
            'ate': self.ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'Double Machine Learning (Simple)',
            'model_type': 'random_forest'
        }


if __name__ == "__main__":
    print("="*70)
    print("DEMO: Double Machine Learning")
    print("="*70)
    
    from data.generators import SimulationConfig, generate_customer_data, BehaviorSimulator
    
    # Generate biased data
    config = SimulationConfig(n_customers=3000, random_seed=42)
    customers = generate_customer_data(config)
    simulator = BehaviorSimulator(customers, config)
    
    biased_data = simulator.simulate_experiment(
        treatment_assignment='biased_activity',
        discount_amount=0.20
    )
    
    confounders = ['activity_score', 'tenure_months', 'prev_purchases', 'account_value']
    
    # Try EconML version
    if ECONML_AVAILABLE:
        print("\nUsing EconML (recommended):")
        print("-"*70)
        
        dml = DoubleMachineLearning(
            confounders=confounders,
            model_type='linear'
        )
        
        dml.fit(biased_data)
        results = dml.estimate_ate(biased_data)  # Pass data here
        
        print(dml.report(results))
        
        # Estimate heterogeneous effects
        cate = dml.estimate_cate(biased_data)
        print(f"\nHeterogeneous Effects:")
        print(f"  Min CATE: {cate.min():+.4f}")
        print(f"  Mean CATE: {cate.mean():+.4f}")
        print(f"  Max CATE: {cate.max():+.4f}")
        
    else:
        print("\nUsing Simple DML (EconML not available):")
        print("-"*70)
        
        dml_simple = SimpleDoubleMachineLearning(confounders=confounders)
        results = dml_simple.fit_and_estimate(biased_data)
        
        print(f"\nATE: {results['ate']:+.4f}")
        print(f"95% CI: [{results['ci_lower']:+.4f}, {results['ci_upper']:+.4f}]")
    
    # Compare to ground truth
    true_ate = biased_data['true_ITE'].mean()
    naive_ate = (
        biased_data[biased_data['treated']==1]['purchased'].mean() -
        biased_data[biased_data['treated']==0]['purchased'].mean()
    )
    
    print(f"\n" + "="*70)
    print("COMPARISON:")
    print(f"  Naive ATE:  {naive_ate:+.4f} (BIASED)")
    print(f"  DML ATE:    {results['ate']:+.4f}")
    print(f"  True ATE:   {true_ate:+.4f}")
    print(f"\n  DML Error:   {abs(results['ate'] - true_ate):.4f}")
    print(f"  Naive Error: {abs(naive_ate - true_ate):.4f}")
    print("="*70)