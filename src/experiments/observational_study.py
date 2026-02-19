"""
Observational Study Framework
==============================

Tools for analyzing non-randomized data where treatment assignment
is confounded with other variables.

This sets up the problems that Module 3 (causal inference) will solve!
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ObservationalStudyConfig:
    """Configuration for observational study"""
    
    treatment_col: str = 'treated'
    outcome_col: str = 'purchased'
    confounders: List[str] = None  # Variables that affect both treatment and outcome
    
    def __post_init__(self):
        if self.confounders is None:
            self.confounders = []


class ObservationalStudy:
    """
    Framework for observational studies.
    
    Key difference from A/B tests:
    - Treatment NOT randomly assigned
    - Need to account for confounding
    - Naive comparison is BIASED
    
    This class helps DIAGNOSE the problem. Module 3 will SOLVE it.
    """
    
    def __init__(self, config: ObservationalStudyConfig):
        self.config = config
    
    def check_balance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Check covariate balance between treatment and control groups.
        
        In randomized experiments, groups should be balanced.
        In observational data, imbalance indicates confounding!
        
        Returns:
            DataFrame showing mean values by treatment group
        """
        if not self.config.confounders:
            print("⚠️  No confounders specified. Add them to config.confounders")
            return pd.DataFrame()
        
        balance_check = data.groupby(self.config.treatment_col)[
            self.config.confounders
        ].mean()
        
        # Calculate standardized mean differences
        control_means = balance_check.loc[0]
        treatment_means = balance_check.loc[1]
        
        # Pooled standard deviation
        control_std = data[data[self.config.treatment_col]==0][self.config.confounders].std()
        treatment_std = data[data[self.config.treatment_col]==1][self.config.confounders].std()
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        
        smd = (treatment_means - control_means) / pooled_std
        
        balance_df = pd.DataFrame({
            'Control Mean': control_means,
            'Treatment Mean': treatment_means,
            'Difference': treatment_means - control_means,
            'SMD': smd,
            'Imbalanced': (np.abs(smd) > 0.1).map({True: '⚠️ YES', False: '✅ No'})
        })
        
        return balance_df
    
    def naive_comparison(self, data: pd.DataFrame) -> Dict:
        """
        Perform naive comparison (what most people do).
        
        WARNING: This is BIASED in observational studies!
        Use this as a baseline to compare against causal methods.
        """
        control = data[data[self.config.treatment_col] == 0]
        treatment = data[data[self.config.treatment_col] == 1]
        
        control_outcome = control[self.config.outcome_col].mean()
        treatment_outcome = treatment[self.config.outcome_col].mean()
        
        absolute_diff = treatment_outcome - control_outcome
        relative_diff = absolute_diff / control_outcome if control_outcome > 0 else np.inf
        
        return {
            'control_mean': control_outcome,
            'treatment_mean': treatment_outcome,
            'absolute_difference': absolute_diff,
            'relative_difference': relative_diff,
            'relative_difference_pct': relative_diff * 100,
            'n_control': len(control),
            'n_treatment': len(treatment),
            'warning': '⚠️  This estimate is likely BIASED due to confounding!'
        }
    
    def visualize_confounding(self, data: pd.DataFrame, confounder: str):
        """
        Create a simple analysis showing confounding structure.
        
        Shows:
        1. Treatment assignment by confounder
        2. Outcome by confounder  
        3. The confounding relationship
        """
        # Bin the confounder for visualization
        data = data.copy()
        bins = pd.qcut(data[confounder], q=5, duplicates='drop', labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # Treatment rate by confounder level
        treatment_by_conf = data.groupby(bins)[self.config.treatment_col].mean()
        
        # Outcome rate by confounder level
        outcome_by_conf = data.groupby(bins)[self.config.outcome_col].mean()
        
        # Correlation check
        from scipy.stats import pearsonr
        
        conf_treatment_corr, conf_treatment_p = pearsonr(
            data[confounder], 
            data[self.config.treatment_col]
        )
        
        conf_outcome_corr, conf_outcome_p = pearsonr(
            data[confounder],
            data[self.config.outcome_col]
        )
        
        report = []
        report.append(f"\nCONFOUNDING ANALYSIS: {confounder}")
        report.append("="*60)
        
        report.append(f"\n1. {confounder} → Treatment Assignment")
        report.append(f"   Correlation: {conf_treatment_corr:.3f} (p={conf_treatment_p:.4f})")
        report.append(f"   Treatment rate by {confounder}:")
        for q, rate in treatment_by_conf.items():
            report.append(f"     {q}: {rate:.3f}")
        
        report.append(f"\n2. {confounder} → Outcome")
        report.append(f"   Correlation: {conf_outcome_corr:.3f} (p={conf_outcome_p:.4f})")
        report.append(f"   Outcome rate by {confounder}:")
        for q, rate in outcome_by_conf.items():
            report.append(f"     {q}: {rate:.3f}")
        
        report.append(f"\n3. Confounding Check")
        if abs(conf_treatment_corr) > 0.1 and abs(conf_outcome_corr) > 0.1:
            report.append(f"   ⚠️  CONFOUNDING DETECTED!")
            report.append(f"   {confounder} affects BOTH treatment and outcome.")
            report.append(f"   Naive comparison will be BIASED.")
        else:
            report.append(f"   ✅ No strong confounding detected.")
        
        return "\n".join(report)
    
    def generate_report(self, data: pd.DataFrame) -> str:
        """
        Generate comprehensive observational study report.
        """
        report = []
        report.append("="*70)
        report.append("OBSERVATIONAL STUDY REPORT")
        report.append("="*70)
        
        # Sample size
        report.append(f"\nSAMPLE SIZE")
        report.append(f"  Total: {len(data):,}")
        report.append(f"  Treatment: {data[self.config.treatment_col].sum():,}")
        report.append(f"  Control: {(~data[self.config.treatment_col].astype(bool)).sum():,}")
        
        # Covariate balance
        if self.config.confounders:
            report.append(f"\nCOVARIATE BALANCE")
            balance = self.check_balance(data)
            report.append(balance.to_string())
            
            # Summarize imbalance
            n_imbalanced = (balance['Imbalanced'] == '⚠️ YES').sum()
            if n_imbalanced > 0:
                report.append(f"\n⚠️  {n_imbalanced} variable(s) show imbalance (|SMD| > 0.1)")
                report.append("   This indicates CONFOUNDING!")
            else:
                report.append("\n✅ All covariates appear balanced")
        
        # Naive comparison
        report.append(f"\nNAIVE COMPARISON (LIKELY BIASED!)")
        naive = self.naive_comparison(data)
        report.append(f"  Control mean:   {naive['control_mean']:.3f}")
        report.append(f"  Treatment mean: {naive['treatment_mean']:.3f}")
        report.append(f"  Difference:     {naive['absolute_difference']:+.3f}")
        report.append(f"  Relative:       {naive['relative_difference_pct']:+.2f}%")
        report.append(f"\n  {naive['warning']}")
        
        report.append("\n" + "="*70)
        report.append("NEXT STEPS: Use causal inference methods (Module 3)")
        report.append("  - Propensity Score Matching")
        report.append("  - Double Machine Learning")
        report.append("  - Instrumental Variables")
        report.append("="*70)
        
        return "\n".join(report)


class DifferenceInDifferences:
    """
    Difference-in-Differences (DiD) for quasi-experimental designs.
    
    Used when:
    - Treatment applied to one group at a specific time
    - Have pre- and post-treatment data for both groups
    - Assume parallel trends
    
    Example: New policy in California but not Texas
    """
    
    def __init__(
        self,
        treatment_col: str = 'treated',
        outcome_col: str = 'outcome',
        time_col: str = 'period',
        treatment_period: int = None
    ):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.time_col = time_col
        self.treatment_period = treatment_period
    
    def estimate_did(self, data: pd.DataFrame) -> Dict:
        """
        Estimate DiD treatment effect.
        
        Formula:
        DiD = (Treatment_After - Treatment_Before) - (Control_After - Control_Before)
        """
        # Split data
        pre = data[data[self.time_col] < self.treatment_period]
        post = data[data[self.time_col] >= self.treatment_period]
        
        # Calculate means
        treated_pre = pre[pre[self.treatment_col]==1][self.outcome_col].mean()
        treated_post = post[post[self.treatment_col]==1][self.outcome_col].mean()
        control_pre = pre[pre[self.treatment_col]==0][self.outcome_col].mean()
        control_post = post[post[self.treatment_col]==0][self.outcome_col].mean()
        
        # DiD estimate
        treatment_diff = treated_post - treated_pre
        control_diff = control_post - control_pre
        did_estimate = treatment_diff - control_diff
        
        return {
            'treated_pre': treated_pre,
            'treated_post': treated_post,
            'control_pre': control_pre,
            'control_post': control_post,
            'treatment_diff': treatment_diff,
            'control_diff': control_diff,
            'did_estimate': did_estimate,
        }
    
    def check_parallel_trends(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Check parallel trends assumption (pre-treatment periods only).
        
        Groups should have similar trends before treatment.
        """
        pre = data[data[self.time_col] < self.treatment_period]
        
        trends = pre.groupby([self.time_col, self.treatment_col])[
            self.outcome_col
        ].mean().unstack()
        
        trends.columns = ['Control', 'Treatment']
        return trends


if __name__ == "__main__":
    # Demo: Observational study with confounding
    print("="*70)
    print("DEMO: Observational Study Analysis")
    print("="*70)
    
    # Generate biased data
    np.random.seed(42)
    n = 5000
    
    # Activity is a confounder
    activity = np.random.uniform(0, 100, n)
    
    # High activity → more likely to get treatment
    treatment_prob = 0.2 + 0.6 * (activity / 100)
    treated = np.random.binomial(1, treatment_prob)
    
    # High activity → more likely to purchase (even without treatment)
    base_purchase_prob = 0.1 + 0.3 * (activity / 100)
    
    # TRUE treatment effect is only +5%
    true_effect = 0.05
    purchase_prob = base_purchase_prob + (true_effect * treated)
    purchased = np.random.binomial(1, purchase_prob)
    
    data = pd.DataFrame({
        'customer_id': range(n),
        'activity_score': activity,
        'treated': treated,
        'purchased': purchased
    })
    
    # Analyze
    config = ObservationalStudyConfig(
        treatment_col='treated',
        outcome_col='purchased',
        confounders=['activity_score']
    )
    
    study = ObservationalStudy(config)
    
    # Check balance
    print("\nCOVARIATE BALANCE CHECK:")
    balance = study.check_balance(data)
    print(balance)
    
    # Naive comparison
    print("\n" + "="*70)
    naive = study.naive_comparison(data)
    print(f"NAIVE ESTIMATE (BIASED!):")
    print(f"  Treatment mean: {naive['treatment_mean']:.3f}")
    print(f"  Control mean:   {naive['control_mean']:.3f}")
    print(f"  Naive effect:   {naive['relative_difference_pct']:+.2f}%")
    print(f"\n  TRUE effect:    +5.00%")
    print(f"  BIAS:           {naive['relative_difference_pct'] - 5:.2f} percentage points!")
    
    # Confounding analysis
    print("\n" + study.visualize_confounding(data, 'activity_score'))
    
    print("\n" + "="*70)
    print("This is why we need causal inference! (Module 3)")
    print("="*70)
