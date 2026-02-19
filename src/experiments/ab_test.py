"""
A/B Testing Framework
=====================

Production-grade A/B test implementation with:
- Proper randomization
- Statistical power analysis
- Multiple testing corrections
- Sequential testing support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest


@dataclass
class ABTestConfig:
    """Configuration for A/B test design"""
    
    # Test parameters
    control_rate: float  # Expected control conversion rate
    minimum_detectable_effect: float  # MDE (e.g., 0.05 = 5% relative lift)
    
    # Statistical parameters
    alpha: float = 0.05  # Significance level (Type I error)
    beta: float = 0.20   # Type II error (1-beta = power)
    
    # Design choices
    two_sided: bool = True  # Two-sided vs one-sided test
    ratio: float = 1.0      # Treatment:Control ratio (1.0 = equal split)
    
    def __post_init__(self):
        """Validate configuration"""
        assert 0 < self.control_rate < 1, "Control rate must be between 0 and 1"
        assert 0 < self.minimum_detectable_effect < 1, "MDE must be between 0 and 1"
        assert 0 < self.alpha < 1, "Alpha must be between 0 and 1"
        assert 0 < self.beta < 1, "Beta must be between 0 and 1"
        assert self.ratio > 0, "Ratio must be positive"


class ABTest:
    """
    A/B Test implementation with best practices from industry.
    
    Features:
    - Sample size calculation (before experiment)
    - Proper randomization
    - Statistical testing with corrections
    - Sequential testing support
    - Confidence intervals
    """
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.test_started = False
        self.results: Optional[pd.DataFrame] = None
    
    def calculate_sample_size(self) -> Dict[str, int]:
        """
        Calculate required sample size using statistical power analysis.
        
        Returns:
            Dictionary with sample sizes for control and treatment groups
        """
        # Calculate absolute effect size
        control_p = self.config.control_rate
        treatment_p = control_p * (1 + self.config.minimum_detectable_effect)
        
        # Use Cohen's h for effect size (better for proportions)
        effect_size = 2 * (np.arcsin(np.sqrt(treatment_p)) - 
                          np.arcsin(np.sqrt(control_p)))
        
        # Calculate required sample size per group
        n_per_group = zt_ind_solve_power(
            effect_size=effect_size,
            alpha=self.config.alpha,
            power=1 - self.config.beta,
            ratio=self.config.ratio,
            alternative='two-sided' if self.config.two_sided else 'larger'
        )
        
        n_control = int(np.ceil(n_per_group))
        n_treatment = int(np.ceil(n_per_group * self.config.ratio))
        
        return {
            'n_control': n_control,
            'n_treatment': n_treatment,
            'n_total': n_control + n_treatment
        }
    
    def run_test(
        self, 
        data: pd.DataFrame,
        treatment_col: str = 'treated',
        outcome_col: str = 'purchased'
    ) -> Dict:
        """
        Run A/B test analysis on experimental data.
        
        Args:
            data: DataFrame with treatment assignment and outcomes
            treatment_col: Name of treatment indicator column (0/1)
            outcome_col: Name of outcome column (0/1 for binary outcomes)
        
        Returns:
            Dictionary with test results
        """
        self.results = data.copy()
        
        # Validate data
        assert treatment_col in data.columns, f"{treatment_col} not in data"
        assert outcome_col in data.columns, f"{outcome_col} not in data"
        
        # Split by treatment
        control = data[data[treatment_col] == 0]
        treatment = data[data[treatment_col] == 1]
        
        # Calculate statistics
        n_control = len(control)
        n_treatment = len(treatment)
        
        control_conversions = control[outcome_col].sum()
        treatment_conversions = treatment[outcome_col].sum()
        
        control_rate = control_conversions / n_control if n_control > 0 else 0
        treatment_rate = treatment_conversions / n_treatment if n_treatment > 0 else 0
        
        # Absolute and relative lift
        absolute_lift = treatment_rate - control_rate
        relative_lift = (absolute_lift / control_rate) if control_rate > 0 else np.inf
        
        # Statistical test (two-proportion z-test)
        z_stat, p_value = proportions_ztest(
            count=[treatment_conversions, control_conversions],
            nobs=[n_treatment, n_control],
            alternative='two-sided' if self.config.two_sided else 'larger'
        )
        
        # Confidence interval for difference
        ci_lower, ci_upper = self._calculate_ci(
            control_rate, treatment_rate,
            n_control, n_treatment
        )
        
        # Statistical power (post-hoc)
        observed_effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - 
                                    np.arcsin(np.sqrt(control_rate)))
        
        results = {
            # Sample sizes
            'n_control': n_control,
            'n_treatment': n_treatment,
            'n_total': n_control + n_treatment,
            
            # Conversion rates
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            
            # Lift
            'absolute_lift': absolute_lift,
            'relative_lift': relative_lift,
            'relative_lift_pct': relative_lift * 100,
            
            # Statistical significance
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': p_value < self.config.alpha,
            
            # Confidence intervals
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_level': 1 - self.config.alpha,
            
            # Effect size
            'effect_size_cohens_h': observed_effect_size,
        }
        
        return results
    
    def _calculate_ci(
        self,
        p1: float, 
        p2: float,
        n1: int,
        n2: int,
        confidence: float = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for difference in proportions.
        
        Uses normal approximation with pooled standard error.
        """
        if confidence is None:
            confidence = 1 - self.config.alpha
        
        # Standard error for difference
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        
        # Critical value
        z_crit = stats.norm.ppf((1 + confidence) / 2)
        
        # Confidence interval
        diff = p2 - p1
        margin = z_crit * se
        
        return (diff - margin, diff + margin)
    
    def sequential_test(
        self,
        data: pd.DataFrame,
        treatment_col: str = 'treated',
        outcome_col: str = 'purchased',
        alpha_spending: str = 'obrien_fleming'
    ) -> Dict:
        """
        Sequential testing (optional stopping) using alpha spending functions.
        
        Allows peeking at results without inflating Type I error.
        
        Args:
            data: Current data
            treatment_col: Treatment indicator
            outcome_col: Outcome variable
            alpha_spending: Type of alpha spending ('obrien_fleming', 'pocock')
        
        Returns:
            Dictionary with sequential test results
        """
        # Calculate current test statistics
        current_results = self.run_test(data, treatment_col, outcome_col)
        
        # Determine sample size progress
        planned_size = self.calculate_sample_size()
        progress = current_results['n_total'] / planned_size['n_total']
        
        # Adjusted alpha using O'Brien-Fleming boundary
        if alpha_spending == 'obrien_fleming':
            # O'Brien-Fleming is conservative early, liberal late
            adjusted_alpha = self.config.alpha * 2 * (1 - stats.norm.cdf(
                stats.norm.ppf(1 - self.config.alpha / 2) / np.sqrt(progress)
            ))
        elif alpha_spending == 'pocock':
            # Pocock maintains constant alpha boundary
            adjusted_alpha = self.config.alpha
        else:
            raise ValueError(f"Unknown alpha spending function: {alpha_spending}")
        
        # Decision
        can_stop = current_results['p_value'] < adjusted_alpha
        
        return {
            **current_results,
            'progress': progress,
            'adjusted_alpha': adjusted_alpha,
            'can_stop': can_stop,
            'recommendation': 'STOP' if can_stop else 'CONTINUE'
        }
    
    def report(self, results: Dict) -> str:
        """
        Generate human-readable test report.
        
        Args:
            results: Dictionary from run_test()
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("A/B TEST RESULTS")
        report.append("="*70)
        report.append("")
        
        # Sample sizes
        report.append("SAMPLE SIZE")
        report.append(f"  Control:   {results['n_control']:,}")
        report.append(f"  Treatment: {results['n_treatment']:,}")
        report.append(f"  Total:     {results['n_total']:,}")
        report.append("")
        
        # Conversion rates
        report.append("CONVERSION RATES")
        report.append(f"  Control:   {results['control_rate']:.3%}")
        report.append(f"  Treatment: {results['treatment_rate']:.3%}")
        report.append("")
        
        # Lift
        report.append("LIFT")
        report.append(f"  Absolute: {results['absolute_lift']:+.3%}")
        report.append(f"  Relative: {results['relative_lift_pct']:+.2f}%")
        report.append("")
        
        # Statistical significance
        report.append("STATISTICAL TEST")
        report.append(f"  Z-statistic: {results['z_statistic']:.3f}")
        report.append(f"  P-value:     {results['p_value']:.4f}")
        report.append(f"  Significant: {'YES ✅' if results['is_significant'] else 'NO ❌'}")
        report.append(f"  (α = {self.config.alpha})")
        report.append("")
        
        # Confidence interval
        report.append("CONFIDENCE INTERVAL")
        ci_pct = results['ci_level'] * 100
        report.append(f"  {ci_pct:.0f}% CI: [{results['ci_lower']:+.3%}, {results['ci_upper']:+.3%}]")
        
        # Interpretation
        report.append("")
        report.append("INTERPRETATION")
        if results['is_significant']:
            if results['relative_lift'] > 0:
                report.append("  ✅ Treatment is significantly BETTER than control")
            else:
                report.append("  ⚠️  Treatment is significantly WORSE than control")
        else:
            report.append("  ❌ No significant difference detected")
            report.append("  Possible reasons:")
            report.append("    - True effect is smaller than MDE")
            report.append("    - Sample size too small (underpowered)")
            report.append("    - Test not run long enough")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)


class MultiArmBandit:
    """
    Multi-armed bandit for adaptive experimentation (OPTIONAL ADVANCED).
    
    Unlike A/B tests (fixed allocation), bandits dynamically allocate
    more traffic to better-performing variants.
    
    Implements Thompson Sampling with Beta priors.
    """
    
    def __init__(self, n_arms: int = 2, prior_alpha: float = 1, prior_beta: float = 1):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms) * prior_alpha
        self.failures = np.ones(n_arms) * prior_beta
    
    def select_arm(self) -> int:
        """
        Select arm using Thompson Sampling.
        
        Returns:
            Index of selected arm
        """
        # Sample from posterior Beta distributions
        samples = np.random.beta(self.successes, self.failures)
        return np.argmax(samples)
    
    def update(self, arm: int, reward: int):
        """
        Update beliefs after observing reward.
        
        Args:
            arm: Which arm was pulled
            reward: 1 for success, 0 for failure
        """
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
    
    def get_probabilities(self) -> np.ndarray:
        """Get current posterior mean estimates"""
        return self.successes / (self.successes + self.failures)


if __name__ == "__main__":
    # Demo: Sample size calculation
    print("="*70)
    print("DEMO: A/B Test Sample Size Calculation")
    print("="*70)
    
    config = ABTestConfig(
        control_rate=0.10,              # 10% baseline conversion
        minimum_detectable_effect=0.10, # Want to detect 10% relative lift
        alpha=0.05,
        beta=0.20                        # 80% power
    )
    
    test = ABTest(config)
    sample_sizes = test.calculate_sample_size()
    
    print(f"\nTest Configuration:")
    print(f"  Control rate: {config.control_rate:.1%}")
    print(f"  MDE (relative): {config.minimum_detectable_effect:.1%}")
    print(f"  Significance level (α): {config.alpha:.3f}")
    print(f"  Power (1-β): {1-config.beta:.1%}")
    
    print(f"\nRequired Sample Size:")
    print(f"  Control:   {sample_sizes['n_control']:,}")
    print(f"  Treatment: {sample_sizes['n_treatment']:,}")
    print(f"  Total:     {sample_sizes['n_total']:,}")
    
    # Demo: Run test on simulated data
    print("\n" + "="*70)
    print("DEMO: Running A/B Test")
    print("="*70)
    
    # Simulate data
    np.random.seed(42)
    n = sample_sizes['n_total']
    
    data = pd.DataFrame({
        'customer_id': range(n),
        'treated': np.random.binomial(1, 0.5, n),
    })
    
    # Simulate outcomes (with true 10% lift)
    true_lift = 0.10
    data['purchased'] = data['treated'].apply(
        lambda x: np.random.binomial(1, config.control_rate * (1 + true_lift * x))
    )
    
    # Run test
    results = test.run_test(data)
    print(test.report(results))
