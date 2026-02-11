"""
Configuration for the CausalCommerce Simulation Engine

This module defines the parameters that control customer generation,
behavior simulation, and causal structures.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class SimulationConfig:
    """Master configuration for the simulation"""
    
    # Population size
    n_customers: int = 10000
    random_seed: int = 42
    
    # Simulation period
    n_time_periods: int = 12  # e.g., 12 months
    
    # Customer segment proportions (must sum to 1.0)
    segment_proportions: Dict[str, float] = None
    
    # Treatment parameters
    discount_amounts: List[float] = None  # e.g., [0.0, 0.10, 0.20, 0.30]
    
    def __post_init__(self):
        if self.segment_proportions is None:
            self.segment_proportions = {
                'loyalists': 0.20,      # Buy regardless of discount
                'persuadables': 0.35,   # Respond positively to discounts
                'sleeping_dogs': 0.15,  # Annoyed by marketing (churn)
                'lost_causes': 0.30     # Never buy
            }
        
        if self.discount_amounts is None:
            self.discount_amounts = [0.0, 0.10, 0.20, 0.30]
    
    def validate(self):
        """Validate configuration parameters"""
        assert abs(sum(self.segment_proportions.values()) - 1.0) < 1e-6, \
            "Segment proportions must sum to 1.0"
        assert all(0 <= p <= 1 for p in self.segment_proportions.values()), \
            "Segment proportions must be between 0 and 1"
        assert self.n_customers > 0, "n_customers must be positive"
        return True


@dataclass
class CustomerSegmentParams:
    """Parameters defining a customer segment's behavior"""
    
    # Base purchase probability (without treatment)
    base_purchase_prob: float
    
    # Treatment effect parameters
    discount_sensitivity: float  # How much discount increases purchase prob
    ad_sensitivity: float        # Response to advertising
    
    # Churn parameters
    base_churn_prob: float
    treatment_churn_effect: float  # Negative = treatment increases churn
    
    # Lifetime value
    avg_purchase_value: float
    purchase_value_std: float
    
    # Activity level (affects selection into treatment in biased scenarios)
    avg_activity_score: float  # 0-100, higher = more engaged


# Define default parameters for each segment
SEGMENT_PARAMS = {
    'loyalists': CustomerSegmentParams(
        base_purchase_prob=0.80,       # High baseline
        discount_sensitivity=0.05,     # Barely responds to discount
        ad_sensitivity=0.02,           # Doesn't need ads
        base_churn_prob=0.02,          # Very loyal
        treatment_churn_effect=0.0,    # Neutral to marketing
        avg_purchase_value=150.0,
        purchase_value_std=30.0,
        avg_activity_score=85.0        # Very active
    ),
    
    'persuadables': CustomerSegmentParams(
        base_purchase_prob=0.30,       # Moderate baseline
        discount_sensitivity=0.40,     # Strong discount response!
        ad_sensitivity=0.25,           # Ads help
        base_churn_prob=0.05,
        treatment_churn_effect=0.0,
        avg_purchase_value=100.0,
        purchase_value_std=25.0,
        avg_activity_score=50.0        # Moderate activity
    ),
    
    'sleeping_dogs': CustomerSegmentParams(
        base_purchase_prob=0.15,       # Low baseline
        discount_sensitivity=-0.10,    # NEGATIVE: discount annoys them!
        ad_sensitivity=-0.15,          # Ads annoy them
        base_churn_prob=0.08,
        treatment_churn_effect=-0.15,  # Marketing increases churn!
        avg_purchase_value=80.0,
        purchase_value_std=20.0,
        avg_activity_score=25.0        # Low activity
    ),
    
    'lost_causes': CustomerSegmentParams(
        base_purchase_prob=0.05,       # Very low baseline
        discount_sensitivity=0.02,     # Barely budges
        ad_sensitivity=0.01,
        base_churn_prob=0.20,          # High churn
        treatment_churn_effect=-0.05,
        avg_purchase_value=60.0,
        purchase_value_std=15.0,
        avg_activity_score=15.0        # Inactive
    )
}