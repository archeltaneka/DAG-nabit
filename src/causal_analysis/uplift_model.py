"""
Uplift Modeling
===============

Predict individual-level treatment effects for optimal targeting.

The idea:
Instead of predicting "who will buy?", predict "who will buy BECAUSE of the treatment?"

Key insight: We want to target PERSUADABLES, not LOYALISTS!
- Loyalists: Buy anyway (waste of discount)
- Persuadables: Only buy with treatment (TARGET THESE!)
- Lost Causes: Won't buy even with treatment (don't bother)
- Sleeping Dogs: Treatment hurts (AVOID!)

Uses: Uber's CausalML library
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Try to import CausalML
try:
    from causalml.inference.meta import BaseXRegressor, BaseRRegressor
    from causalml.metrics import plot_gain, qini_score
    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False
    print("⚠️  CausalML not installed. Install with: pip install causalml")


class UpliftModel:
    """
    Uplift modeling for heterogeneous treatment effect estimation.
    
    Predicts individual-level treatment effects (uplift).
    Used for optimal customer targeting.
    
    Meta-learners available:
    - X-Learner: Best for imbalanced treatment/control
    - R-Learner: Similar to DML but with single model
    - S-Learner: Single model (baseline)
    - T-Learner: Separate models for treated/control
    """
    
    def __init__(
        self,
        treatment_col: str = 'treated',
        outcome_col: str = 'purchased',
        features: List[str] = None,
        meta_learner: str = 'x',
        base_model: str = 'xgboost'
    ):
        """
        Args:
            treatment_col: Treatment indicator
            outcome_col: Outcome variable
            features: Features for prediction
            meta_learner: 'x' (X-Learner) or 'r' (R-Learner)
            base_model: Base ML model ('xgboost', 'random_forest')
        """
        if not CAUSALML_AVAILABLE:
            raise ImportError("CausalML is required. Install with: pip install causalml")
        
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.features = features or []
        self.meta_learner = meta_learner
        self.base_model = base_model
        
        self.model = None
        self.uplift_scores = None
    
    def fit(self, data: pd.DataFrame) -> 'UpliftModel':
        """
        Fit the uplift model.
        
        Returns:
            self (for method chaining)
        """
        X = data[self.features].values
        T = data[self.treatment_col].values
        Y = data[self.outcome_col].values
        
        # Choose base learner
        if self.base_model == 'xgboost':
            from xgboost import XGBRegressor
            learner = XGBRegressor(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            learner = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Choose meta-learner
        if self.meta_learner == 'x':
            self.model = BaseXRegressor(learner=learner)
        elif self.meta_learner == 'r':
            self.model = BaseRRegressor(learner=learner)
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner}")
        
        # Fit
        self.model.fit(X, T, Y)
        
        return self
    
    def predict_uplift(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict individual treatment effects (uplift).
        
        Returns:
            Array of predicted uplifts for each customer
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        X = data[self.features].values
        
        # Predict uplift
        self.uplift_scores = self.model.predict(X).flatten()
        
        return self.uplift_scores
    
    def segment_customers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Segment customers based on predicted uplift.
        
        Returns:
            DataFrame with customer segments
        """
        if self.uplift_scores is None:
            self.predict_uplift(data)
        
        data_copy = data.copy()
        data_copy['predicted_uplift'] = self.uplift_scores
        
        # Segment by uplift quartile
        data_copy['uplift_quartile'] = pd.qcut(
            self.uplift_scores,
            q=4,
            labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)']
        )
        
        # Identify segments (simplified)
        conditions = [
            data_copy['predicted_uplift'] > 0.10,   # Strong positive
            data_copy['predicted_uplift'] > 0.02,   # Weak positive
            data_copy['predicted_uplift'] > -0.02,  # Neutral
            data_copy['predicted_uplift'] <= -0.02  # Negative
        ]
        segments = ['Persuadable', 'Weak Responder', 'Neutral', 'Sleeping Dog']
        
        data_copy['predicted_segment'] = np.select(conditions, segments, default='Unknown')
        
        return data_copy
    
    def evaluate(self, data: pd.DataFrame) -> Dict:
        """
        Evaluate model using various metrics.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.uplift_scores is None:
            self.predict_uplift(data)
        
        # Get ground truth if available
        if 'true_ITE' in data.columns:
            true_uplift = data['true_ITE'].values
            
            # Correlation with ground truth
            from scipy.stats import pearsonr, spearmanr
            pearson_corr, pearson_p = pearsonr(self.uplift_scores, true_uplift)
            spearman_corr, spearman_p = spearmanr(self.uplift_scores, true_uplift)
            
            # RMSE
            rmse = np.sqrt(np.mean((self.uplift_scores - true_uplift) ** 2))
            
            return {
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'rmse': rmse,
                'has_ground_truth': True
            }
        else:
            return {
                'has_ground_truth': False,
                'message': 'No ground truth available for evaluation'
            }
    
    def calculate_targeting_value(
        self,
        data: pd.DataFrame,
        budget_pct: float = 0.5,
        cost_per_treatment: float = 10,
        value_per_conversion: float = 100
    ) -> Dict:
        """
        Calculate business value of uplift-based targeting.
        
        Args:
            budget_pct: Fraction of customers to target
            cost_per_treatment: Cost to deliver treatment
            value_per_conversion: Revenue per conversion
        
        Returns:
            Comparison of random vs. uplift-based targeting
        """
        if self.uplift_scores is None:
            self.predict_uplift(data)
        
        n_target = int(len(data) * budget_pct)
        
        # Strategy 1: Random targeting
        random_sample = data.sample(n=n_target, random_state=42)
        
        if 'true_ITE' in random_sample.columns:
            random_incremental = random_sample['true_ITE'].sum()
        else:
            # Estimate from uplift scores
            random_incremental = self.uplift_scores[random_sample.index].sum()
        
        random_cost = n_target * cost_per_treatment
        random_profit = (random_incremental * value_per_conversion) - random_cost
        
        # Strategy 2: Uplift-based targeting
        data_with_uplift = data.copy()
        data_with_uplift['predicted_uplift'] = self.uplift_scores
        
        uplift_sample = data_with_uplift.nlargest(n_target, 'predicted_uplift')
        
        if 'true_ITE' in uplift_sample.columns:
            uplift_incremental = uplift_sample['true_ITE'].sum()
        else:
            uplift_incremental = self.uplift_scores[uplift_sample.index].sum()
        
        uplift_cost = n_target * cost_per_treatment
        uplift_profit = (uplift_incremental * value_per_conversion) - uplift_cost
        
        return {
            'n_targeted': n_target,
            'random_incremental_conversions': random_incremental,
            'random_profit': random_profit,
            'uplift_incremental_conversions': uplift_incremental,
            'uplift_profit': uplift_profit,
            'improvement': uplift_profit - random_profit,
            'improvement_pct': (uplift_profit - random_profit) / abs(random_profit) * 100 if random_profit != 0 else 0
        }
    
    def report(self, evaluation: Dict, targeting: Dict) -> str:
        """Generate formatted report"""
        report = []
        report.append("="*70)
        report.append("UPLIFT MODELING RESULTS")
        report.append("="*70)
        
        report.append(f"\nModel: {self.meta_learner.upper()}-Learner")
        report.append(f"Base Model: {self.base_model}")
        
        if evaluation.get('has_ground_truth'):
            report.append(f"\nModel Performance (vs Ground Truth):")
            report.append(f"  Pearson Correlation: {evaluation['pearson_correlation']:.3f}")
            report.append(f"  Spearman Correlation: {evaluation['spearman_correlation']:.3f}")
            report.append(f"  RMSE: {evaluation['rmse']:.4f}")
        
        report.append(f"\nTargeting Value Analysis:")
        report.append(f"  Customers targeted: {targeting['n_targeted']:,}")
        report.append(f"\n  Random Targeting:")
        report.append(f"    Incremental conversions: {targeting['random_incremental_conversions']:.2f}")
        report.append(f"    Profit: ${targeting['random_profit']:,.2f}")
        report.append(f"\n  Uplift-Based Targeting:")
        report.append(f"    Incremental conversions: {targeting['uplift_incremental_conversions']:.2f}")
        report.append(f"    Profit: ${targeting['uplift_profit']:,.2f}")
        report.append(f"\n  Improvement:")
        report.append(f"    Profit gain: ${targeting['improvement']:,.2f}")
        report.append(f"    ROI improvement: {targeting['improvement_pct']:+.1f}%")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


if __name__ == "__main__":
    print("="*70)
    print("DEMO: Uplift Modeling")
    print("="*70)
    
    if not CAUSALML_AVAILABLE:
        print("\n⚠️  CausalML not installed.")
        print("Install with: pip install causalml xgboost")
        exit(1)
    
    from data.generators import SimulationConfig, generate_customer_data, BehaviorSimulator
    
    # Generate data with heterogeneous effects
    config = SimulationConfig(n_customers=5000, random_seed=42)
    customers = generate_customer_data(config)
    simulator = BehaviorSimulator(customers, config)
    
    # Random experiment to train on
    experiment = simulator.simulate_experiment(
        treatment_assignment='random',
        treatment_probability=0.5,
        discount_amount=0.20
    )
    
    # Add true ITE for evaluation
    experiment['true_ITE'] = experiment['discount_effect'] * 0.20
    
    # Train uplift model
    features = ['activity_score', 'tenure_months', 'prev_purchases', 'account_value', 'email_engagement_rate']
    
    uplift = UpliftModel(
        features=features,
        meta_learner='x',
        base_model='xgboost'
    )
    
    print("\nTraining uplift model...")
    uplift.fit(experiment)
    
    # Predict
    uplift_scores = uplift.predict_uplift(experiment)
    
    print(f"\nPredicted Uplift Distribution:")
    print(f"  Min:  {uplift_scores.min():+.4f}")
    print(f"  Mean: {uplift_scores.mean():+.4f}")
    print(f"  Max:  {uplift_scores.max():+.4f}")
    
    # Segment customers
    segmented = uplift.segment_customers(experiment)
    
    print(f"\nPredicted Customer Segments:")
    segment_counts = segmented['predicted_segment'].value_counts()
    for segment, count in segment_counts.items():
        pct = count / len(segmented) * 100
        print(f"  {segment:20s}: {count:5,} ({pct:5.1f}%)")
    
    # Evaluate
    evaluation = uplift.evaluate(experiment)
    
    print(f"\nModel Accuracy:")
    print(f"  Correlation with true uplift: {evaluation['pearson_correlation']:.3f}")
    print(f"  RMSE: {evaluation['rmse']:.4f}")
    
    # Calculate targeting value
    targeting = uplift.calculate_targeting_value(experiment, budget_pct=0.5)
    
    print(uplift.report(evaluation, targeting))
    
    # Compare predicted vs true segments
    if 'segment' in experiment.columns:
        print("\nPredicted vs True Segments:")
        confusion = pd.crosstab(
            experiment['segment'],
            segmented['predicted_segment'],
            normalize='index'
        )
        print(confusion.round(2))
