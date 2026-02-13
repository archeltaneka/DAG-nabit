"""
Propensity Score Matching
==========================

Classic causal inference method for removing selection bias.

The idea:
1. Estimate probability of treatment (propensity score) from observables
2. Match treated customers to similar control customers
3. Compare outcomes only among matched pairs
4. This removes confounding!

Reference: Rosenbaum & Rubin (1983)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats


class PropensityScoreMatcher:
    """
    Propensity Score Matching for causal inference.
    
    Removes selection bias by matching treated and control units
    with similar propensity scores (probability of treatment).
    
    Key assumptions:
    - Unconfoundedness: All confounders are observed
    - Common support: Overlap in propensity scores
    - Stable Unit Treatment Value Assumption (SUTVA)
    """
    
    def __init__(
        self,
        treatment_col: str = 'treated',
        outcome_col: str = 'purchased',
        confounders: List[str] = None,
        matching_method: str = 'nearest',
        caliper: float = None
    ):
        """
        Args:
            treatment_col: Name of treatment indicator (0/1)
            outcome_col: Name of outcome variable
            confounders: List of confounder variable names
            matching_method: 'nearest' or 'radius'
            caliper: Maximum distance for matching (in std dev of propensity score)
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounders = confounders or []
        self.matching_method = matching_method
        self.caliper = caliper
        
        self.propensity_model = None
        self.propensity_scores = None
        self.matches = None
    
    def fit_propensity_model(self, data: pd.DataFrame) -> np.ndarray:
        """
        Estimate propensity scores using logistic regression.
        
        Returns:
            Array of propensity scores (probability of treatment)
        """
        X = data[self.confounders].values
        y = data[self.treatment_col].values
        
        # Fit logistic regression
        self.propensity_model = LogisticRegression(max_iter=1000, random_state=42)
        self.propensity_model.fit(X, y)
        
        # Predict propensity scores
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        return self.propensity_scores
    
    def check_common_support(self, data: pd.DataFrame) -> Dict:
        """
        Check overlap in propensity score distributions.
        
        Common support is essential for valid matching.
        If treated and control groups don't overlap, matching fails!
        """
        if self.propensity_scores is None:
            raise ValueError("Must fit propensity model first")
        
        ps = self.propensity_scores
        treated = data[self.treatment_col] == 1
        
        ps_treated = ps[treated]
        ps_control = ps[~treated]
        
        # Calculate overlap
        min_treated = ps_treated.min()
        max_treated = ps_treated.max()
        min_control = ps_control.min()
        max_control = ps_control.max()
        
        # Common support region
        common_min = max(min_treated, min_control)
        common_max = min(max_treated, max_control)
        
        # Proportion in common support
        in_support_treated = ((ps_treated >= common_min) & (ps_treated <= common_max)).mean()
        in_support_control = ((ps_control >= common_min) & (ps_control <= common_max)).mean()
        
        return {
            'common_support_min': common_min,
            'common_support_max': common_max,
            'treated_in_support': in_support_treated,
            'control_in_support': in_support_control,
            'has_overlap': common_min < common_max
        }
    
    def match(self, data: pd.DataFrame, replace: bool = False) -> pd.DataFrame:
        """
        Perform matching between treated and control units.
        
        Args:
            data: DataFrame with treatment, outcome, and confounders
            replace: Whether to sample control units with replacement
        
        Returns:
            DataFrame with matched pairs
        """
        if self.propensity_scores is None:
            self.fit_propensity_model(data)
        
        data = data.copy()
        data['propensity_score'] = self.propensity_scores
        
        # Split into treated and control
        treated = data[data[self.treatment_col] == 1].copy()
        control = data[data[self.treatment_col] == 0].copy()
        
        if self.matching_method == 'nearest':
            matches = self._nearest_neighbor_matching(
                treated, control, replace=replace
            )
        elif self.matching_method == 'radius':
            matches = self._radius_matching(treated, control)
        else:
            raise ValueError(f"Unknown matching method: {self.matching_method}")
        
        self.matches = matches
        return matches
    
    def _nearest_neighbor_matching(
        self,
        treated: pd.DataFrame,
        control: pd.DataFrame,
        replace: bool = False
    ) -> pd.DataFrame:
        """
        Match each treated unit to nearest control unit(s) by propensity score.
        """
        # Fit nearest neighbors on control propensity scores
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(control[['propensity_score']].values)
        
        # Find matches
        distances, indices = nn.kneighbors(treated[['propensity_score']].values)
        
        # Apply caliper if specified
        if self.caliper is not None:
            ps_std = self.propensity_scores.std()
            max_distance = self.caliper * ps_std
            valid_matches = distances.flatten() <= max_distance
        else:
            valid_matches = np.ones(len(treated), dtype=bool)
        
        # Build matched dataset
        matched_pairs = []
        used_controls = set()
        
        for i, (treated_idx, control_idx, dist, is_valid) in enumerate(
            zip(treated.index, indices.flatten(), distances.flatten(), valid_matches)
        ):
            if not is_valid:
                continue
            
            control_idx_actual = control.index[control_idx]
            
            # Skip if control already used (when replace=False)
            if not replace and control_idx_actual in used_controls:
                continue
            
            used_controls.add(control_idx_actual)
            
            # Add matched pair
            matched_pairs.append({
                'treated_id': treated_idx,
                'control_id': control_idx_actual,
                'match_distance': dist,
                'treated_outcome': treated.loc[treated_idx, self.outcome_col],
                'control_outcome': control.loc[control_idx_actual, self.outcome_col],
                'propensity_score_treated': treated.loc[treated_idx, 'propensity_score'],
                'propensity_score_control': control.loc[control_idx_actual, 'propensity_score']
            })
        
        return pd.DataFrame(matched_pairs)
    
    def _radius_matching(
        self,
        treated: pd.DataFrame,
        control: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Match treated units to all controls within a radius (caliper).
        """
        if self.caliper is None:
            raise ValueError("Caliper must be specified for radius matching")
        
        ps_std = self.propensity_scores.std()
        max_distance = self.caliper * ps_std
        
        matched_pairs = []
        
        for treated_idx, treated_row in treated.iterrows():
            treated_ps = treated_row['propensity_score']
            
            # Find all controls within radius
            distances = np.abs(control['propensity_score'] - treated_ps)
            within_radius = control[distances <= max_distance]
            
            # Add all matches
            for control_idx, control_row in within_radius.iterrows():
                matched_pairs.append({
                    'treated_id': treated_idx,
                    'control_id': control_idx,
                    'match_distance': abs(treated_ps - control_row['propensity_score']),
                    'treated_outcome': treated_row[self.outcome_col],
                    'control_outcome': control_row[self.outcome_col],
                    'propensity_score_treated': treated_ps,
                    'propensity_score_control': control_row['propensity_score']
                })
        
        return pd.DataFrame(matched_pairs)
    
    def estimate_ate(self) -> Dict:
        """
        Estimate Average Treatment Effect (ATE) from matched pairs.
        
        Returns:
            Dictionary with ATE estimate and statistics
        """
        if self.matches is None or len(self.matches) == 0:
            raise ValueError("Must perform matching first")
        
        # Individual treatment effects
        self.matches['ite'] = (
            self.matches['treated_outcome'] - self.matches['control_outcome']
        )
        
        # Average Treatment Effect
        ate = self.matches['ite'].mean()
        ate_se = self.matches['ite'].std() / np.sqrt(len(self.matches))
        
        # Confidence interval
        z_crit = stats.norm.ppf(0.975)  # 95% CI
        ci_lower = ate - z_crit * ate_se
        ci_upper = ate + z_crit * ate_se
        
        # T-test
        t_stat, p_value = stats.ttest_1samp(self.matches['ite'], 0)
        
        return {
            'ate': ate,
            'ate_se': ate_se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_matches': len(self.matches),
            'is_significant': p_value < 0.05
        }
    
    def check_balance_after_matching(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Check covariate balance after matching.
        
        Good matching should eliminate differences in confounders.
        """
        if self.matches is None:
            raise ValueError("Must perform matching first")
        
        # Get matched sample
        treated_ids = self.matches['treated_id'].values
        control_ids = self.matches['control_id'].values
        
        matched_treated = data.loc[treated_ids, self.confounders]
        matched_control = data.loc[control_ids, self.confounders]
        
        # Calculate standardized mean differences
        balance = []
        for var in self.confounders:
            mean_t = matched_treated[var].mean()
            mean_c = matched_control[var].mean()
            std_t = matched_treated[var].std()
            std_c = matched_control[var].std()
            
            pooled_std = np.sqrt((std_t**2 + std_c**2) / 2)
            smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0
            
            balance.append({
                'variable': var,
                'treated_mean': mean_t,
                'control_mean': mean_c,
                'smd': smd,
                'balanced': abs(smd) < 0.1
            })
        
        return pd.DataFrame(balance)
    
    def report(self, results: Dict) -> str:
        """Generate formatted report"""
        report = []
        report.append("="*70)
        report.append("PROPENSITY SCORE MATCHING RESULTS")
        report.append("="*70)
        
        report.append(f"\nMatched Pairs: {results['n_matches']:,}")
        
        report.append(f"\nAverage Treatment Effect (ATE):")
        report.append(f"  Estimate: {results['ate']:+.4f}")
        report.append(f"  Std Error: {results['ate_se']:.4f}")
        report.append(f"  95% CI: [{results['ci_lower']:+.4f}, {results['ci_upper']:+.4f}]")
        
        report.append(f"\nStatistical Test:")
        report.append(f"  T-statistic: {results['t_statistic']:.3f}")
        report.append(f"  P-value: {results['p_value']:.4f}")
        report.append(f"  Significant: {'YES ✅' if results['is_significant'] else 'NO ❌'}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo with biased data
    print("="*70)
    print("DEMO: Propensity Score Matching")
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
    
    # Naive comparison
    naive_ate = (
        biased_data[biased_data['treated']==1]['purchased'].mean() -
        biased_data[biased_data['treated']==0]['purchased'].mean()
    )
    
    print(f"\nNaive ATE (BIASED): {naive_ate:+.4f}")
    
    # Propensity score matching
    confounders = ['activity_score', 'tenure_months', 'prev_purchases', 'account_value']
    
    psm = PropensityScoreMatcher(
        confounders=confounders,
        matching_method='nearest',
        caliper=0.1
    )
    
    # Fit propensity model
    ps_scores = psm.fit_propensity_model(biased_data)
    
    # Check common support
    support = psm.check_common_support(biased_data)
    print(f"\nCommon Support Check:")
    print(f"  Treated in support: {support['treated_in_support']:.1%}")
    print(f"  Control in support: {support['control_in_support']:.1%}")
    
    # Match
    matches = psm.match(biased_data)
    print(f"\nMatched {len(matches):,} pairs")
    
    # Estimate ATE
    results = psm.estimate_ate()
    print(psm.report(results))
    
    # Check balance
    balance = psm.check_balance_after_matching(biased_data)
    print("\nCovariate Balance After Matching:")
    print(balance)
    
    # Compare to ground truth
    true_ate = biased_data['true_ITE'].mean() if 'true_ITE' in biased_data else None
    if true_ate is not None:
        print(f"\n" + "="*70)
        print("COMPARISON TO GROUND TRUTH:")
        print(f"  Naive ATE:  {naive_ate:+.4f} (BIASED)")
        print(f"  PSM ATE:    {results['ate']:+.4f}")
        print(f"  True ATE:   {true_ate:+.4f}")
        print(f"\n  PSM Error:   {abs(results['ate'] - true_ate):.4f}")
        print(f"  Naive Error: {abs(naive_ate - true_ate):.4f}")
        print(f"\n  PSM is {abs(naive_ate - true_ate) / abs(results['ate'] - true_ate):.1f}x more accurate!")
        print("="*70)