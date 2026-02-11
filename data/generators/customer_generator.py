"""
Customer Generator - Creates synthetic customers with hidden causal properties

This module generates individual customers with:
- Observable features (demographics, activity)
- Hidden propensities (true treatment effects)
- Segment membership (the ground truth we'll try to recover)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .config import SimulationConfig, SEGMENT_PARAMS


class CustomerGenerator:
    """
    Generates a synthetic customer population with hidden causal structures.
    
    Key Design:
    - Each customer has a TRUE segment (loyalist, persuadable, etc.)
    - This segment determines their hidden propensities
    - Observable features are correlated with (but don't perfectly reveal) segment
    - This creates the challenge: can causal inference recover true effects?
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        np.random.seed(config.random_seed)
        self.rng = np.random.default_rng(config.random_seed)
    
    def generate_population(self) -> pd.DataFrame:
        """
        Generate complete customer population
        
        Returns:
            DataFrame with columns:
            - customer_id
            - segment (TRUE segment - hidden in real world!)
            - observable features (age, tenure, activity_score, etc.)
            - hidden propensities (purchase_propensity, discount_effect, etc.)
        """
        n = self.config.n_customers
        
        # Step 1: Assign segments based on proportions
        segments = self._assign_segments(n)
        
        # Step 2: Generate observable features
        features = self._generate_observable_features(segments)
        
        # Step 3: Generate hidden propensities (the causal truth!)
        propensities = self._generate_hidden_propensities(segments)
        
        # Step 4: Combine into dataframe
        df = pd.DataFrame({
            'customer_id': range(n),
            'segment': segments,
            **features,
            **propensities
        })
        
        return df
    
    def _assign_segments(self, n: int) -> np.ndarray:
        """Randomly assign customers to segments based on proportions"""
        segments = []
        proportions = self.config.segment_proportions
        
        for segment, prop in proportions.items():
            n_segment = int(n * prop)
            segments.extend([segment] * n_segment)
        
        # Handle rounding
        while len(segments) < n:
            segments.append(self.rng.choice(list(proportions.keys())))
        
        segments = np.array(segments[:n])
        self.rng.shuffle(segments)
        return segments
    
    def _generate_observable_features(self, segments: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate observable customer features.
        
        Key: Features are CORRELATED with segment but don't perfectly reveal it.
        This mimics reality: we can observe behavior, but not internal preferences.
        """
        n = len(segments)
        
        # Activity score - correlated with segment
        activity_scores = np.zeros(n)
        for i, segment in enumerate(segments):
            params = SEGMENT_PARAMS[segment]
            # Add noise so it's not deterministic
            activity_scores[i] = self.rng.normal(
                params.avg_activity_score, 
                15.0  # standard deviation
            )
        activity_scores = np.clip(activity_scores, 0, 100)
        
        # Age - slight correlation with segment
        base_ages = {
            'loyalists': 45,
            'persuadables': 35,
            'sleeping_dogs': 50,
            'lost_causes': 28
        }
        ages = np.array([
            self.rng.normal(base_ages[seg], 12) 
            for seg in segments
        ])
        ages = np.clip(ages, 18, 80).astype(int)
        
        # Tenure (months as customer) - correlated with loyalty
        base_tenure = {
            'loyalists': 36,
            'persuadables': 18,
            'sleeping_dogs': 24,
            'lost_causes': 6
        }
        tenure = np.array([
            self.rng.exponential(base_tenure[seg]) 
            for seg in segments
        ])
        tenure = np.clip(tenure, 0, 120).astype(int)
        
        # Previous purchases (count)
        prev_purchases = np.array([
            self.rng.poisson(
                5 if seg == 'loyalists' else
                2 if seg == 'persuadables' else
                1 if seg == 'sleeping_dogs' else
                0.5
            )
            for seg in segments
        ])
        
        # Account value (historical spending)
        account_values = np.array([
            SEGMENT_PARAMS[seg].avg_purchase_value * prev_purchases[i] * 
            self.rng.uniform(0.8, 1.2)
            for i, seg in enumerate(segments)
        ])
        
        # Email engagement rate (0-1)
        base_engagement = {
            'loyalists': 0.7,
            'persuadables': 0.5,
            'sleeping_dogs': 0.2,
            'lost_causes': 0.3
        }
        email_engagement = np.array([
            np.clip(self.rng.beta(
                base_engagement[seg] * 10,
                (1 - base_engagement[seg]) * 10
            ), 0, 1)
            for seg in segments
        ])
        
        return {
            'age': ages,
            'tenure_months': tenure,
            'activity_score': activity_scores,
            'prev_purchases': prev_purchases,
            'account_value': account_values,
            'email_engagement_rate': email_engagement
        }
    
    def _generate_hidden_propensities(self, segments: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate TRUE causal propensities (hidden in real world!)
        
        These represent the customer's TRUE response to treatment.
        In a real business, we never observe these directly - we only observe
        outcomes. The goal of causal inference is to ESTIMATE these!
        """
        n = len(segments)
        
        propensities = {
            'base_purchase_propensity': np.zeros(n),
            'discount_effect': np.zeros(n),
            'ad_effect': np.zeros(n),
            'churn_propensity': np.zeros(n),
            'treatment_churn_effect': np.zeros(n),
            'avg_purchase_value': np.zeros(n),
        }
        
        for i, segment in enumerate(segments):
            params = SEGMENT_PARAMS[segment]
            
            # Add individual variation within segment
            propensities['base_purchase_propensity'][i] = np.clip(
                self.rng.normal(params.base_purchase_prob, 0.1),
                0, 1
            )
            
            propensities['discount_effect'][i] = self.rng.normal(
                params.discount_sensitivity, 0.05
            )
            
            propensities['ad_effect'][i] = self.rng.normal(
                params.ad_sensitivity, 0.05
            )
            
            propensities['churn_propensity'][i] = np.clip(
                self.rng.normal(params.base_churn_prob, 0.02),
                0, 0.5
            )
            
            propensities['treatment_churn_effect'][i] = self.rng.normal(
                params.treatment_churn_effect, 0.03
            )
            
            propensities['avg_purchase_value'][i] = self.rng.normal(
                params.avg_purchase_value,
                params.purchase_value_std
            )
        
        return propensities


def generate_customer_data(config: SimulationConfig = None) -> pd.DataFrame:
    """
    Convenience function to generate customer population
    
    Args:
        config: SimulationConfig object (uses defaults if None)
    
    Returns:
        DataFrame of customers with observable features and hidden propensities
    """
    if config is None:
        config = SimulationConfig()
    
    config.validate()
    generator = CustomerGenerator(config)
    return generator.generate_population()


if __name__ == "__main__":
    # Example usage
    print("Generating customer population...")
    
    config = SimulationConfig(n_customers=1000)
    customers = generate_customer_data(config)
    
    print(f"\nGenerated {len(customers)} customers")
    print(f"\nSegment distribution:")
    print(customers['segment'].value_counts(normalize=True))
    
    print(f"\nSample of data:")
    print(customers.head(10))
    
    print(f"\nObservable features:")
    observable_cols = ['age', 'tenure_months', 'activity_score', 'prev_purchases', 
                       'account_value', 'email_engagement_rate']
    print(customers[observable_cols].describe())
    
    print(f"\nHidden propensities by segment:")
    for segment in customers['segment'].unique():
        seg_data = customers[customers['segment'] == segment]
        print(f"\n{segment.upper()}:")
        print(f"  Base purchase prob: {seg_data['base_purchase_propensity'].mean():.3f}")
        print(f"  Discount effect: {seg_data['discount_effect'].mean():.3f}")
        print(f"  Churn propensity: {seg_data['churn_propensity'].mean():.3f}")