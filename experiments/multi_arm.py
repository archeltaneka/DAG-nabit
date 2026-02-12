"""
Multi-Arm Experiments
======================

Framework for experiments with multiple treatment variants.

Use cases:
- Dose-response: Testing different discount levels (10%, 20%, 30%)
- Feature variants: Testing different UI designs
- Targeting strategies: Testing different audience segments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from statsmodels.stats.multitest import multipletests


class MultiArmExperiment:
    """
    Multi-arm experiment with multiple testing corrections.
    
    Key challenges:
    1. Multiple comparisons inflate Type I error
    2. Need correction methods (Bonferroni, Holm, etc.)
    3. Sample size increases with number of arms
    """
    
    def __init__(
        self,
        arm_names: List[str],
        control_arm: str = None,
        alpha: float = 0.05
    ):
        """
        Args:
            arm_names: List of treatment arm names
            control_arm: Name of control arm (default: first arm)
            alpha: Overall significance level
        """
        self.arm_names = arm_names
        self.control_arm = control_arm or arm_names[0]
        self.alpha = alpha
        
        assert self.control_arm in arm_names, "Control arm must be in arm_names"
    
    def analyze(
        self,
        data: pd.DataFrame,
        arm_col: str,
        outcome_col: str,
        correction_method: str = 'holm'
    ) -> Dict:
        """
        Analyze multi-arm experiment with multiple testing correction.
        
        Args:
            data: Experiment data
            arm_col: Column indicating arm assignment
            outcome_col: Outcome variable
            correction_method: 'bonferroni', 'holm', 'fdr_bh', or 'none'
        
        Returns:
            Dictionary with results for each comparison
        """
        results = {}
        
        # Get control group
        control_data = data[data[arm_col] == self.control_arm][outcome_col]
        
        # Pairwise comparisons
        comparisons = []
        p_values = []
        
        for arm in self.arm_names:
            if arm == self.control_arm:
                continue
            
            treatment_data = data[data[arm_col] == arm][outcome_col]
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(control_data)-1)*control_data.std()**2 + 
                 (len(treatment_data)-1)*treatment_data.std()**2) /
                (len(control_data) + len(treatment_data) - 2)
            )
            cohens_d = (treatment_data.mean() - control_data.mean()) / pooled_std
            
            comparisons.append(arm)
            p_values.append(p_value)
            
            results[arm] = {
                'control_mean': control_data.mean(),
                'treatment_mean': treatment_data.mean(),
                'difference': treatment_data.mean() - control_data.mean(),
                'relative_lift': ((treatment_data.mean() - control_data.mean()) / 
                                 control_data.mean() * 100),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_control': len(control_data),
                'n_treatment': len(treatment_data)
            }
        
        # Apply multiple testing correction
        if correction_method != 'none' and len(p_values) > 0:
            rejected, adjusted_p, _, _ = multipletests(
                p_values,
                alpha=self.alpha,
                method=correction_method
            )
            
            for i, arm in enumerate(comparisons):
                results[arm]['adjusted_p_value'] = adjusted_p[i]
                results[arm]['significant_corrected'] = rejected[i]
                results[arm]['significant_uncorrected'] = p_values[i] < self.alpha
        
        # Summary statistics
        summary = {
            'n_arms': len(self.arm_names),
            'n_comparisons': len(comparisons),
            'control_arm': self.control_arm,
            'alpha': self.alpha,
            'correction_method': correction_method,
            'arms': results
        }
        
        return summary
    
    def report(self, results: Dict) -> str:
        """Generate formatted report"""
        report = []
        report.append("="*70)
        report.append("MULTI-ARM EXPERIMENT RESULTS")
        report.append("="*70)
        
        report.append(f"\nConfiguration:")
        report.append(f"  Control arm: {results['control_arm']}")
        report.append(f"  Number of arms: {results['n_arms']}")
        report.append(f"  Comparisons: {results['n_comparisons']}")
        report.append(f"  Multiple testing correction: {results['correction_method']}")
        report.append(f"  Overall α: {results['alpha']}")
        
        report.append(f"\nResults by Arm:")
        report.append("-"*70)
        
        for arm, arm_results in results['arms'].items():
            report.append(f"\n{arm} vs {results['control_arm']}:")
            report.append(f"  Mean: {arm_results['treatment_mean']:.4f} vs {arm_results['control_mean']:.4f}")
            report.append(f"  Lift: {arm_results['relative_lift']:+.2f}%")
            report.append(f"  P-value (raw): {arm_results['p_value']:.4f}")
            
            if 'adjusted_p_value' in arm_results:
                report.append(f"  P-value (adjusted): {arm_results['adjusted_p_value']:.4f}")
                sig_marker = "✅" if arm_results['significant_corrected'] else "❌"
                report.append(f"  Significant: {sig_marker} {arm_results['significant_corrected']}")
            
            report.append(f"  Effect size (Cohen's d): {arm_results['cohens_d']:.3f}")
            report.append(f"  Sample size: {arm_results['n_treatment']:,}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


class DoseResponseExperiment:
    """
    Specialized multi-arm experiment for dose-response analysis.
    
    Example: Testing discount levels (0%, 10%, 20%, 30%)
    
    Can fit dose-response curves and find optimal dose.
    """
    
    def __init__(self, doses: List[float], alpha: float = 0.05):
        """
        Args:
            doses: List of dose levels (e.g., [0, 0.1, 0.2, 0.3])
            alpha: Significance level
        """
        self.doses = sorted(doses)
        self.alpha = alpha
    
    def analyze(
        self,
        data: pd.DataFrame,
        dose_col: str,
        outcome_col: str
    ) -> Dict:
        """
        Analyze dose-response relationship.
        
        Returns:
            Dictionary with dose-response curve and optimal dose
        """
        # Calculate mean outcome at each dose
        dose_means = data.groupby(dose_col)[outcome_col].agg(['mean', 'std', 'count'])
        
        # Fit linear dose-response model
        from scipy.stats import linregress
        
        slope, intercept, r_value, p_value, std_err = linregress(
            dose_means.index,
            dose_means['mean']
        )
        
        # Find optimal dose (highest mean)
        optimal_dose = dose_means['mean'].idxmax()
        
        # Test for monotonicity (doses should have increasing/decreasing effect)
        from scipy.stats import spearmanr
        monotonicity_corr, monotonicity_p = spearmanr(
            dose_means.index,
            dose_means['mean']
        )
        
        results = {
            'doses': self.doses,
            'dose_response': dose_means.to_dict('index'),
            'linear_model': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'interpretation': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Flat'
            },
            'optimal_dose': optimal_dose,
            'optimal_response': dose_means.loc[optimal_dose, 'mean'],
            'monotonicity': {
                'correlation': monotonicity_corr,
                'p_value': monotonicity_p,
                'is_monotonic': monotonicity_p < self.alpha
            }
        }
        
        return results
    
    def plot_curve(self, results: Dict) -> str:
        """
        Generate ASCII plot of dose-response curve.
        """
        dose_response = results['dose_response']
        
        report = []
        report.append("\nDOSE-RESPONSE CURVE")
        report.append("="*70)
        
        # Find scale
        max_response = max(dr['mean'] for dr in dose_response.values())
        min_response = min(dr['mean'] for dr in dose_response.values())
        
        for dose, stats in sorted(dose_response.items()):
            # Scale to 50 characters
            scaled = int((stats['mean'] - min_response) / (max_response - min_response + 1e-6) * 50)
            bar = '█' * scaled
            
            optimal = " ← OPTIMAL" if dose == results['optimal_dose'] else ""
            report.append(f"  {dose:5.2f} | {bar} {stats['mean']:.4f}{optimal}")
        
        report.append("\n" + "="*70)
        report.append(f"Linear trend: {results['linear_model']['interpretation']}")
        report.append(f"  Slope: {results['linear_model']['slope']:.4f}")
        report.append(f"  R²: {results['linear_model']['r_squared']:.3f}")
        report.append(f"\nOptimal dose: {results['optimal_dose']:.2f}")
        report.append(f"Expected outcome: {results['optimal_response']:.4f}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo: Multi-arm experiment
    print("="*70)
    print("DEMO: Multi-Arm Experiment with Multiple Testing Correction")
    print("="*70)
    
    np.random.seed(42)
    n_per_arm = 1000
    
    # Simulate data with 4 arms
    arms = ['control', 'variant_A', 'variant_B', 'variant_C']
    true_effects = [0.10, 0.12, 0.11, 0.10]  # Only variant_A has real lift
    
    data_list = []
    for arm, effect in zip(arms, true_effects):
        arm_data = pd.DataFrame({
            'customer_id': range(len(data_list)*n_per_arm, (len(data_list)+1)*n_per_arm),
            'arm': arm,
            'purchased': np.random.binomial(1, effect, n_per_arm)
        })
        data_list.append(arm_data)
    
    data = pd.concat(data_list, ignore_index=True)
    
    # Analyze
    experiment = MultiArmExperiment(
        arm_names=arms,
        control_arm='control',
        alpha=0.05
    )
    
    results_bonf = experiment.analyze(data, 'arm', 'purchased', correction_method='bonferroni')
    results_holm = experiment.analyze(data, 'arm', 'purchased', correction_method='holm')
    
    print("\nWith Bonferroni correction (conservative):")
    print(experiment.report(results_bonf))
    
    print("\n" + "="*70)
    print("With Holm correction (less conservative):")
    print(experiment.report(results_holm))
    
    # Demo: Dose-response
    print("\n\n" + "="*70)
    print("DEMO: Dose-Response Analysis")
    print("="*70)
    
    # Simulate dose-response data
    doses = [0, 0.10, 0.20, 0.30]
    
    dose_data_list = []
    for dose in doses:
        # Response increases with dose but plateaus
        base_rate = 0.10
        response = base_rate + 0.30 * dose - 0.40 * dose**2  # Quadratic (diminishing returns)
        
        dose_df = pd.DataFrame({
            'customer_id': range(len(dose_data_list)*1000, (len(dose_data_list)+1)*1000),
            'discount': dose,
            'purchased': np.random.binomial(1, np.clip(response, 0, 1), 1000)
        })
        dose_data_list.append(dose_df)
    
    dose_data = pd.concat(dose_data_list, ignore_index=True)
    
    # Analyze
    dose_exp = DoseResponseExperiment(doses=doses)
    dose_results = dose_exp.analyze(dose_data, 'discount', 'purchased')
    
    print(dose_exp.plot_curve(dose_results))
    
    print("\n💡 Key insight: Response increases with dose but may plateau!")
    print("   Finding optimal dose maximizes ROI vs. cost.")
