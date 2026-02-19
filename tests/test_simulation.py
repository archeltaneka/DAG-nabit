import pytest
import pandas as pd
import numpy as np

from src.generators.config import SimulationConfig
from src.generators.customer_generator import generate_customer_data

def test_simulation_config_defaults():
    """Test that SimulationConfig initializes with correct defaults"""
    config = SimulationConfig()
    assert config.n_customers == 10000
    assert config.random_seed == 42
    assert abs(sum(config.segment_proportions.values()) - 1.0) < 1e-6

def test_simulation_config_validation():
    """Test validation logic in SimulationConfig"""
    # Valid config
    config = SimulationConfig(n_customers=100)
    assert config.validate() is True
    
    # Invalid proportions
    with pytest.raises(AssertionError):
        bad_config = SimulationConfig(segment_proportions={'a': 0.5, 'b': 0.6})
        bad_config.validate()

    # Invalid n_customers
    with pytest.raises(AssertionError):
        bad_config = SimulationConfig(n_customers=-10)
        bad_config.validate()

def test_customer_generation_shape():
    """Test that generate_customer_data returns dataframe with correct shape"""
    n = 500
    config = SimulationConfig(n_customers=n, random_seed=123)
    df = generate_customer_data(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n
    
    expected_cols = [
        'customer_id', 'segment', 
        'age', 'tenure_months', 'activity_score',
        'base_purchase_propensity', 'discount_effect'
    ]
    for col in expected_cols:
        assert col in df.columns

def test_customer_generation_segments():
    """Test that generated segments match requested proportions roughly"""
    n = 1000
    props = {
        'loyalists': 0.5,
        'persuadables': 0.5
    }
    config = SimulationConfig(n_customers=n, segment_proportions=props)
    df = generate_customer_data(config)
    
    counts = df['segment'].value_counts()
    assert 'loyalists' in counts
    assert 'persuadables' in counts
    # Check if proportions are within 5% tolerance
    assert abs(counts['loyalists'] / n - 0.5) < 0.05
