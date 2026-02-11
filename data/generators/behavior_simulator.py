"""
Behavior Simulator - Simulates customer responses to interventions

This module takes customers (with their hidden propensities) and simulates
what happens when we apply treatments (discounts, ads, etc.).

Key Insight: The simulator uses the HIDDEN propensities to generate outcomes.
This creates realistic data where we observe correlations but need causal
inference to recover the true treatment effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional
from .config import SimulationConfig


class BehaviorSimulator:
    """
    Simulates customer behavior in response to treatments.
    
    The simulator creates a realistic environment where:
    1. We assign treatments (randomly or with bias)
    2. Customers respond based on their hidden propensities
    3. We observe outcomes (purchases, churn, revenue)
    4. Confounders create challenges for naive analysis
    """
    
    def __init__(self, customers: pd.DataFrame, config: SimulationConfig):
        self.customers = customers.copy()
        self.config = config
        self.rng = np.random.default_rng(config.random_seed + 1)
    
    def simulate_experiment(
        self,
        treatment_assignment: Literal['random', 'biased_activity', 'biased_value'] = 'random',
        treatment_probability: float = 0.5,
        discount_amount: float = 0.20
    ) -> pd.DataFrame:
        """
        Simulate a complete marketing experiment.
        
        Args:
            treatment_assignment: How to assign treatment
                - 'random': Pure A/B test (gold standard)
                - 'biased_activity': Higher activity customers more likely to get discount
                - 'biased_value': Higher value customers more likely to get discount
            treatment_probability: Base probability of treatment (for random)
            discount_amount: Size of discount (0.0 to 1.0)
        
        Returns:
            DataFrame with treatment assignments and outcomes
        """
        df = self.customers.copy()
        
        # Step 1: Assign treatment
        df['treated'] = self._assign_treatment(
            df, 
            method=treatment_assignment,
            base_prob=treatment_probability
        )
        df['discount_amount'] = df['treated'] * discount_amount
        
        # Step 2: Simulate outcomes based on treatment and propensities
        outcomes = self._simulate_outcomes(df)
        df = pd.concat([df, outcomes], axis=1)
        
        # Step 3: Calculate derived metrics
        df['revenue'] = df['purchased'] * df['purchase_amount']
        df['gross_profit'] = df['revenue'] - (df['treated'] * 10)  # $10 cost per treatment
        
        return df
    
    def _assign_treatment(
        self, 
        df: pd.DataFrame, 
        method: str, 
        base_prob: float
    ) -> np.ndarray:
        """
        Assign treatment to customers using different strategies.
        
        This is where we can introduce SELECTION BIAS - a key challenge
        that causal inference must overcome!
        """
        n = len(df)
        
        if method == 'random':
            # Pure randomization - no bias
            return self.rng.binomial(1, base_prob, n)
        
        elif method == 'biased_activity':
            # Higher activity customers more likely to get treatment
            # This creates confounding: active customers buy more anyway!
            activity_normalized = df['activity_score'] / 100
            treatment_probs = 0.2 + (0.6 * activity_normalized)  # 20% to 80%
            return self.rng.binomial(1, treatment_probs)
        
        elif method == 'biased_value':
            # Higher value customers more likely to get treatment
            # Marketing team targets "valuable" customers
            value_normalized = (df['account_value'] - df['account_value'].min()) / \
                              (df['account_value'].max() - df['account_value'].min() + 1e-6)
            treatment_probs = 0.2 + (0.6 * value_normalized)
            return self.rng.binomial(1, treatment_probs)
        
        else:
            raise ValueError(f"Unknown treatment assignment method: {method}")
    
    def _simulate_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate customer outcomes based on treatment and hidden propensities.
        
        This is the CORE of the simulation - where causal effects are realized!
        """
        n = len(df)
        outcomes = pd.DataFrame(index=df.index)
        
        # Calculate actual purchase probability for each customer
        # Formula: base_prob + (discount_effect * discount_amount)
        purchase_probs = (
            df['base_purchase_propensity'] + 
            (df['discount_effect'] * df['discount_amount'])
        )
        purchase_probs = np.clip(purchase_probs, 0, 1)
        
        # Simulate purchases
        outcomes['purchased'] = self.rng.binomial(1, purchase_probs)
        
        # Simulate purchase amounts (only for those who purchased)
        outcomes['purchase_amount'] = np.where(
            outcomes['purchased'],
            self.rng.normal(
                df['avg_purchase_value'],
                df['avg_purchase_value'] * 0.2  # 20% std dev
            ),
            0
        )
        outcomes['purchase_amount'] = np.maximum(outcomes['purchase_amount'], 0)
        
        # Simulate churn
        churn_probs = (
            df['churn_propensity'] + 
            (df['treatment_churn_effect'] * df['treated'])
        )
        churn_probs = np.clip(churn_probs, 0, 1)
        outcomes['churned'] = self.rng.binomial(1, churn_probs)
        
        return outcomes
    
    def simulate_time_series(
        self,
        n_periods: int = 12,
        treatment_start_period: int = 6,
        treatment_assignment: str = 'random'
    ) -> pd.DataFrame:
        """
        Simulate behavior over multiple time periods.
        
        Useful for:
        - Difference-in-differences analysis
        - Time-varying treatments
        - Long-term effects
        """
        all_periods = []
        
        for period in range(n_periods):
            df = self.customers.copy()
            df['period'] = period
            
            # Apply treatment only after start period
            if period >= treatment_start_period:
                period_data = self.simulate_experiment(
                    treatment_assignment=treatment_assignment
                )
            else:
                # Pre-treatment period - no one gets treatment
                df['treated'] = 0
                df['discount_amount'] = 0.0
                outcomes = self._simulate_outcomes(df)
                period_data = pd.concat([df, outcomes], axis=1)
                period_data['revenue'] = period_data['purchased'] * period_data['purchase_amount']
                period_data['gross_profit'] = period_data['revenue']
            
            all_periods.append(period_data)
        
        return pd.concat(all_periods, ignore_index=True)


def simulate_scenario(
    scenario: str = 'randomized_ab_test',
    n_customers: int = 5000
) -> pd.DataFrame:
    """
    Convenience function to simulate common business scenarios.
    
    Scenarios:
        'randomized_ab_test': Clean A/B test (no bias)
        'biased_targeting': Marketing targets active customers (confounded)
        'value_based_targeting': Target high-value customers (confounded)
    """
    from .customer_generator import generate_customer_data
    
    config = SimulationConfig(n_customers=n_customers)
    customers = generate_customer_data(config)
    simulator = BehaviorSimulator(customers, config)
    
    if scenario == 'randomized_ab_test':
        return simulator.simulate_experiment(
            treatment_assignment='random',
            treatment_probability=0.5,
            discount_amount=0.20
        )
    
    elif scenario == 'biased_targeting':
        return simulator.simulate_experiment(
            treatment_assignment='biased_activity',
            discount_amount=0.20
        )
    
    elif scenario == 'value_based_targeting':
        return simulator.simulate_experiment(
            treatment_assignment='biased_value',
            discount_amount=0.20
        )
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("CAUSAL COMMERCE - Behavior Simulation Demo")
    print("="*60)
    
    # Scenario 1: Clean randomized experiment
    print("\n" + "="*60)
    print("SCENARIO 1: Randomized A/B Test (No Bias)")
    print("="*60)
    data_random = simulate_scenario('randomized_ab_test', n_customers=2000)
    
    print("\nTreatment assignment:")
    print(data_random['treated'].value_counts())
    
    print("\nNaive comparison (what most people do):")
    naive_results = data_random.groupby('treated').agg({
        'purchased': 'mean',
        'revenue': 'mean',
        'churned': 'mean'
    })
    print(naive_results)
    print(f"\nNaive uplift in purchase rate: {(naive_results.loc[1, 'purchased'] - naive_results.loc[0, 'purchased'])*100:.2f}%")
    
    # Scenario 2: Biased targeting
    print("\n" + "="*60)
    print("SCENARIO 2: Biased Targeting (Active Customers Targeted)")
    print("="*60)
    data_biased = simulate_scenario('biased_targeting', n_customers=2000)
    
    print("\nTreatment assignment by activity level:")
    activity_quintiles = pd.qcut(data_biased['activity_score'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    print(data_biased.groupby(activity_quintiles)['treated'].mean())
    
    print("\nNaive comparison (BIASED!):")
    naive_results_biased = data_biased.groupby('treated').agg({
        'purchased': 'mean',
        'revenue': 'mean'
    })
    print(naive_results_biased)
    print(f"\nNaive uplift in purchase rate: {(naive_results_biased.loc[1, 'purchased'] - naive_results_biased.loc[0, 'purchased'])*100:.2f}%")
    print("\n⚠️  WARNING: This is BIASED because active customers were targeted!")
    print("    Active customers buy more anyway, even without discount.")
    print("    We need causal inference to get the TRUE effect!")
    
    # Show the TRUE effect by segment
    print("\n" + "="*60)
    print("GROUND TRUTH: True Treatment Effects by Segment")
    print("="*60)
    print("\nTrue discount sensitivity by segment:")
    true_effects = data_random.groupby('segment')['discount_effect'].mean()
    print(true_effects.sort_values(ascending=False))
    
    print("\nThis is what we're trying to RECOVER with causal inference!")