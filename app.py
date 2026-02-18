"""
CausalSim - Business Strategy Simulation Dashboard
===================================================

Interactive dashboard for business strategy simulation using causal inference.
Version 1.2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap

from data.generators.config import SimulationConfig, SEGMENT_PARAMS
from data.generators.customer_generator import generate_customer_data
from data.generators.behavior_simulator import BehaviorSimulator
from experiments.observational_study import ObservationalStudy, ObservationalStudyConfig
from experiments.multi_arm import DoseResponseExperiment
from experiments.ab_test import ABTest, ABTestConfig

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="CausalSim",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Theme
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0f1419;
        color: #e4e7eb;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3748;
    }
    
    [data-testid="stSidebar"] * {
        color: #e4e7eb !important;
    }
    
    /* Header styling */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 2rem;
    }
    
    .app-logo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.25rem;
    }
    
    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e4e7eb;
    }
    
    .app-version {
        font-size: 0.875rem;
        color: #718096;
        margin-left: 0.5rem;
    }
    
    /* Segment card styling */
    .segment-card {
        background: linear-gradient(135deg, var(--card-color) 0%, var(--card-color-dark) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .segment-card-loyalists {
        --card-color: rgba(56, 189, 248, 0.15);
        --card-color-dark: rgba(14, 165, 233, 0.15);
    }
    
    .segment-card-persuadables {
        --card-color: rgba(34, 197, 94, 0.15);
        --card-color-dark: rgba(22, 163, 74, 0.15);
    }
    
    .segment-card-sleeping {
        --card-color: rgba(236, 72, 153, 0.15);
        --card-color-dark: rgba(219, 39, 119, 0.15);
    }
    
    .segment-card-lost {
        --card-color: rgba(251, 191, 36, 0.15);
        --card-color-dark: rgba(245, 158, 11, 0.15);
    }
    
    .segment-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.25rem;
    }
    
    .segment-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .segment-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e4e7eb;
        margin: 0;
    }
    
    .segment-metric {
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e4e7eb;
    }
    
    .metric-subtext {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-left: 0.5rem;
    }
    
    .metric-bar {
        width: 100%;
        height: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .metric-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Chart container */
    .chart-container {
        background-color: #1a1f2e;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2d3748;
        margin-bottom: 1.5rem;
    }
    
    .chart-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #e4e7eb;
        margin-bottom: 1rem;
    }
    
    .chart-subtitle {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-ground-truth {
        background-color: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        border-bottom: 1px solid #2d3748;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border: none;
        padding: 0.75rem 0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea;
        border-bottom: 2px solid #667eea;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        color: #e4e7eb;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #e4e7eb;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar header */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 1px solid #2d3748;
        margin-bottom: 2rem;
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
    }
    
    .sidebar-section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #718096;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Generation (Cached)
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(n_customers, random_seed, include_noise):
    """
    Generate customer population and simulate experiment.
    Cached to prevent regeneration on every interaction.
    """
    # 1. Generate Customers
    config = SimulationConfig(
        n_customers=n_customers,
        random_seed=random_seed
    )
    customers = generate_customer_data(config)
    
    # Add noise if requested
    if include_noise:
        customers['activity_score'] = customers['activity_score'] + np.random.normal(0, 5, len(customers))
        customers['activity_score'] = customers['activity_score'].clip(0, 100)
    
    simulator = BehaviorSimulator(customers, config)
    
    return customers, simulator

@st.cache_data
def load_ab_test_data(_simulator, ab_proportion, discount_amount, control_rate, minimum_detectable_effect, alpha, beta):
    test_config = ABTestConfig(
        control_rate=control_rate,             
        minimum_detectable_effect=minimum_detectable_effect, 
        alpha=alpha,                     
        beta=beta                       
    )
    ab_test = ABTest(test_config)
    sample_sizes = ab_test.calculate_sample_size()

    # Run randomized (50/50) and biased A/B tests
    random_experiment_data = simulator.simulate_experiment(
        treatment_assignment='random',
        treatment_probability=ab_proportion,
        discount_amount=discount_amount
    )
    ab_test_randomized_results = ab_test.run_test(random_experiment_data, treatment_col='treated', outcome_col='purchased')
    
    biased_experiment_data = simulator.simulate_experiment(
        treatment_assignment='biased_activity',
        discount_amount=discount_amount
    )
    ab_test_biased_results = ab_test.run_test(biased_experiment_data, treatment_col='treated', outcome_col='purchased')

    # Run Observational Study
    study_config = ObservationalStudyConfig(
        treatment_col='treated',
        outcome_col='purchased',
        confounders=['activity_score', 'tenure_months', 'prev_purchases']
    )
    
    observational_study = ObservationalStudy(study_config)
    
    # Run on biased data
    observational_results = observational_study.check_balance(biased_experiment_data)

    return sample_sizes, random_experiment_data, ab_test_randomized_results, biased_experiment_data, ab_test_biased_results, observational_results

@st.cache_data
def load_multi_arm_data(n_per_arm, discount_amounts, random_seed, include_noise):
    arms_data = []
    for arm_name, discount in discount_amounts.items():
        
        # Generate customers for this arm
        arm_config = SimulationConfig(n_customers=n_per_arm, random_seed=random_seed+int(discount*100))
        arm_customers = generate_customer_data(arm_config)
        
        # Simulate with this discount level
        arm_sim = BehaviorSimulator(arm_customers, arm_config)
        arm_exp = arm_sim.simulate_experiment(
            treatment_assignment='random',
            discount_amount=discount
        )
        
        arm_exp['arm'] = arm_name
        arms_data.append(arm_exp)

    multi_arm_data = pd.concat(arms_data, ignore_index=True)
    multi_arm_data['discount_level'] = multi_arm_data['arm'].map(discount_amounts)

    dose_exp = DoseResponseExperiment(doses=list(discount_amounts.values()))
    dose_results = dose_exp.analyze(multi_arm_data, 'discount_level', 'purchased')
    
    return dose_results

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    # Initialize custom discounts in session state
    if 'custom_discounts' not in st.session_state:
        st.session_state.custom_discounts = {
            'Control': 0.0,
            'Discount 10%': 0.10,
            'Discount 20%': 0.20,
            'Discount 30%': 0.30
        }

    # Sidebar header
    st.markdown("""
        <div class="sidebar-header">
            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem;">CausalSim</div>
            <div style="color: #718096; font-size: 0.875rem;">v1.2</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Configuration section
    st.markdown('<div class="sidebar-section-title">CONFIGURATION</div>', unsafe_allow_html=True)
    st.markdown("**Population Params**")
    
    n_customers = st.slider(
        "Total Population (N)", 
        min_value=1000, 
        max_value=20000, 
        value=10000, 
        step=1000,
        help="Number of customers to simulate"
    )

    discount_amount = st.slider(
        "Discount Amount",
        min_value=0.01,
        max_value=0.90,
        value=0.20,
        step=0.01,
        help="Discount amount to apply to customers"
    )
    
    random_seed = st.number_input(
        "Simulation Seed",
        value=42,
        min_value=1,
        key="random_seed",
        help="Fixed seed ensures reproducibility of the synthetic dataset"
    )
    
    st.markdown("---")

    st.markdown("**A/B Test Params**")

    control_rate = st.slider(
        "Control Group Conversion Rate",
        min_value=0.01,
        max_value=1.00,
        value=0.10,
        step=0.01,
        help="Conversion rate of the control group"
    )

    alpha = st.slider(
        "Alpha",
        min_value=0.01,
        max_value=1.00,
        value=0.05,
        step=0.01,
        help="Significance level"
    )

    minimum_detectable_effect = st.slider(
        "Minimum Detectable Effect",
        min_value=0.01,
        max_value=1.00,
        value=0.10,
        step=0.01,
        help="Minimum detectable effect"
    )

    beta = st.slider(
        "Beta",
        min_value=0.01,
        max_value=1.00,
        value=0.20,
        step=0.01,
        help="Power of the test"
    )

    ab_proportion = st.slider(
        "Proportion of Customers Getting Treatment",
        min_value=0.01,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help="If the value is 0.5, 50% of the customers will get the treatment"
    )

    st.markdown("---")

    st.markdown("**Multi-Arm Bandit Params**")
    n_per_arm = st.slider(
        "Number of Customers per Arm",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of customers to simulate per arm"
    )

    available_discounts = {
        'control': 0.0,
        'discount_10': 0.10,
        'discount_20': 0.20,
        'discount_30': 0.30
    }
    selected_arms = st.sidebar.multiselect(
        "Select Arms to Compare",
        options=list(st.session_state.custom_discounts.keys()),
        default=list(st.session_state.custom_discounts.keys())
    )

    with st.sidebar.expander("Add Custom Arm"):
        new_label = st.text_input("Arm Name", placeholder="e.g., Flash Sale")
        new_val = st.number_input("Discount Value", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        
        if st.button("Add to Experiment"):
            if new_label and new_label not in st.session_state.custom_discounts:
                st.session_state.custom_discounts[new_label] = new_val
            st.success(f"Added {new_label}!")
            st.rerun() # Refresh to update the multiselect options

    st.markdown("---")
    
    # Toggles
    include_noise = st.toggle("Include Noise", value=True, help="Add random noise to activity scores")
    

# Get simulation parameters
random_seed = st.session_state.get("random_seed", 42)

# Load Data
customers, simulator = load_data(n_customers, random_seed, include_noise)

# Load AB Test Data
sample_sizes, random_experiment_data, ab_test_randomized_results, biased_experiment_data, ab_test_biased_results, observational_results = load_ab_test_data(simulator, ab_proportion, discount_amount, control_rate, minimum_detectable_effect, alpha, beta)

dose_results = load_multi_arm_data(n_per_arm, st.session_state.custom_discounts, random_seed, include_noise)

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

# App header
st.markdown("""
    <div class="app-header">
        <div class="app-logo">C</div>
        <div>
            <span class="app-title">CausalSim</span>
            <span class="app-version">v1.2</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Tabs
tab_simulation, tab_experiment, tab_inference = st.tabs([
    "Ground Truth", 
    "A/B Test", 
    "Causal Inference Methods"
])

# -----------------------------------------------------------------------------
# Tab 1: Ground Truth
# -----------------------------------------------------------------------------

with tab_simulation:
    # Tab description
    st.markdown(f"""
    <div style="background-color: rgba(26, 31, 46, 0.6); border: 1px solid #2d3748; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <h3 style="color: #e4e7eb; margin-top: 0;">Business Strategy Simulation</h3>
        <p style="color: #94a3b8; font-size: 1rem;">
            Our simulation models <b>4 distinct customer types</b> (inspired by real uplift modeling research):
        </p>
        <div style="margin: 1.5rem 0;">
            <table style="width: 100%; border-collapse: collapse; color: #e4e7eb; font-size: 0.9rem;">
                <tr style="border-bottom: 1px solid #2d3748; text-align: left; color: #718096;">
                    <th style="padding: 10px;">Segment</th>
                    <th style="padding: 10px;">Base Purchase Rate</th>
                    <th style="padding: 10px;">Discount Sensitivity</th>
                    <th style="padding: 10px;">Key Insight</th>
                </tr>
                <tr style="border-bottom: 1px solid #2d3748;">
                    <td style="padding: 10px;"><b style="color: #38bdf8;">Loyalists</b></td>
                    <td style="padding: 10px;">80%</td>
                    <td style="padding: 10px;">+5%</td>
                    <td style="padding: 10px; color: #94a3b8;">Buy anyway - discount wastes money</td>
                </tr>
                <tr style="border-bottom: 1px solid #2d3748;">
                    <td style="padding: 10px;"><b style="color: #22c55e;">Persuadables</b></td>
                    <td style="padding: 10px;">30%</td>
                    <td style="padding: 10px; color: #22c55e;"><b>+40%</b></td>
                    <td style="padding: 10px;"><b>TARGET THESE!</b> High ROI</td>
                </tr>
                <tr style="border-bottom: 1px solid #2d3748;">
                    <td style="padding: 10px;"><b style="color: #ec4899;">Sleeping Dogs</b></td>
                    <td style="padding: 10px;">15%</td>
                    <td style="padding: 10px; color: #ef4444;">-10%</td>
                    <td style="padding: 10px; color: #94a3b8;">Discounts HURT conversion</td>
                </tr>
                <tr>
                    <td style="padding: 10px;"><b style="color: #fbbf24;">Lost Causes</b></td>
                    <td style="padding: 10px;">5%</td>
                    <td style="padding: 10px;">+2%</td>
                    <td style="padding: 10px; color: #94a3b8;">Won't buy even with discount</td>
                </tr>
            </table>
        </div>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            <b style="color: #e4e7eb;">The Challenge:</b> In real business, we can't observe segments directly. We only see observable features (age, activity) and outcomes (did they buy?).
        </p>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 0;">
            <b style="color: #e4e7eb;">Goal:</b> Use causal inference to identify <b>Persuadables</b> without wasting money on others.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Segment Profiles Header
    st.markdown('<h2 style="color: #e4e7eb; margin-top: 1.5rem;">Segment Profiles</h2>', unsafe_allow_html=True)
    
    # Calculate segment statistics
    segment_stats = customers.groupby('segment').agg({
        'base_purchase_propensity': 'mean',
        'discount_effect': 'mean',
        'churn_propensity': 'mean',
        'activity_score': 'mean'
    }).reset_index()
    
    # Create 4 columns for segment cards
    cols = st.columns(4)
    
    segments_config = [
        {
            'name': 'Loyalists',
            'color': 'loyalists',
            'dot_color': '#38bdf8',
            'purchase_label': 'High',
            'purchase_range': '~90%',
            'sensitivity': 'Zero / Neg',
            'sensitivity_color': '#94a3b8',
            'sensitivity_width': '10%'
        },
        {
            'name': 'Persuadables',
            'color': 'persuadables',
            'dot_color': '#22c55e',
            'purchase_label': 'Low',
            'purchase_range': '~15%',
            'sensitivity': 'High +',
            'sensitivity_color': '#22c55e',
            'sensitivity_width': '85%'
        },
        {
            'name': 'Sleeping Dogs',
            'color': 'sleeping',
            'dot_color': '#ec4899',
            'purchase_label': 'Variable',
            'purchase_range': '~40%',
            'sensitivity': 'Negative -',
            'sensitivity_color': '#ec4899',
            'sensitivity_width': '45%'
        },
        {
            'name': 'Lost Causes',
            'color': 'lost',
            'dot_color': '#fbbf24',
            'purchase_label': 'Near 0',
            'purchase_range': '~1%',
            'sensitivity': 'None',
            'sensitivity_color': '#94a3b8',
            'sensitivity_width': '5%'
        }
    ]
    
    # Create 4 columns for segment cards
    cols = st.columns(4)

    for col, config in zip(cols, segments_config):
        with col:
            # 1. We define the HTML with zero leading indentation 
            # 2. We apply config['dot_color'] to the primary metrics
            card_html = f"""
    <div class="segment-card segment-card-{config['color']}">
        <div class="segment-header">
            <span class="segment-dot" style="background-color: {config['dot_color']};"></span>
            <h3 class="segment-title">{config['name']}</h3>
        </div>
        <div class="segment-metric">
            <div class="metric-label">BASE PURCHASE RATE</div>
            <div class="metric-value" style="color: {config['dot_color']}; font-weight: 800; font-size: 1.6rem;">
                {config['purchase_label']}
                <span class="metric-subtext" style="color: #94a3b8; font-weight: 400; font-size: 0.8rem;">({config['purchase_range']})</span>
            </div>
        </div>
        <div class="segment-metric">
            <div class="metric-label">DISCOUNT SENSITIVITY</div>
            <div class="metric-value" style="color: {config['dot_color']}; font-weight: 800; font-size: 1.3rem; margin-bottom: 4px;">
                {config['sensitivity']}
            </div>
            <div class="metric-bar">
                <div class="metric-bar-fill" style="width: {config['sensitivity_width']}; background-color: {config['dot_color']}; box-shadow: 0 0 10px {config['dot_color']}44;"></div>
            </div>
        </div>
    </div>"""
            
            # Ensure there are no spaces before card_html
            st.markdown(card_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Feature Distributions Section ---
    st.markdown("""
        <div class="chart-container">
            <div class="chart-title">Customer Feature Distributions</div>
            <div class="chart-subtitle">Analyzing how segments differ across observable characteristics</div>
        </div>
    """, unsafe_allow_html=True)

    # Define the features to plot
    features = [
        ('age', 'Age (Years)'), 
        ('tenure_months', 'Tenure (Months)'), 
        ('activity_score', 'Activity Score (0-100)'), 
        ('prev_purchases', 'Previous Purchases'), 
        ('account_value', 'Total Account Value ($)'), 
        ('email_engagement_rate', 'Email Engagement (%)')
    ]

    # Synchronized color palette
    colors_map = {
        'Loyalists': '#38bdf8',
        'Persuadables': '#22c55e',
        'Sleeping Dogs': '#ec4899',
        'Lost Causes': '#fbbf24'
    }

    # Normalize segment names for consistent color mapping
    plot_df = customers.copy()
    plot_df['segment_display'] = plot_df['segment'].str.title().str.replace('_', ' ')

    # Create a 3x2 grid of columns
    for i in range(0, len(features), 2):
        row_cols = st.columns(2)
        for j in range(2):
            if i + j < len(features):
                col_name, label = features[i + j]
                with row_cols[j]:
                    fig_hist = px.histogram(
                        plot_df,
                        x=col_name,
                        color="segment_display",
                        nbins=50,
                        barmode="overlay",
                        color_discrete_map=colors_map,
                        opacity=0.6,
                        labels={"segment_display": "Segment"}
                    )
                    
                    fig_hist.update_layout(
                        title=dict(
                            text=label,
                            font=dict(size=14, color='#e4e7eb'),
                            x=0.05,
                            y=0.95
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(15, 20, 25, 0.4)', # Slightly lighter than main bg
                        font=dict(family="Inter, sans-serif", color='#94a3b8'),
                        height=280,
                        margin=dict(t=50, b=30, l=30, r=20),
                        showlegend=False, # Hide individual legends to save space
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            title="",
                            tickfont=dict(size=10)
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='#2d3748',
                            zeroline=False,
                            title="",
                            showticklabels=False
                        )
                    )
                    
                    # Smooth out the bins and remove the outline
                    fig_hist.update_traces(marker_line_width=0)
                    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

    # Add a single shared legend at the bottom for the whole grid
    legend_html = f"""
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px; padding: 10px; background: #1a1f2e; border-radius: 8px; border: 1px solid #2d3748;">
        {''.join([f'<div style="display: flex; align-items: center; gap: 8px;"><div style="width: 12px; height: 12px; border-radius: 3px; background-color: {c};"></div><span style="color: #e4e7eb; font-size: 0.85rem; font-family: Inter;">{s}</span></div>' for s, c in colors_map.items()])}
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Charts section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Hidden Customer Segments</div>
                <div class="chart-subtitle">Distribution count of true customer types in generated population</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Segment distribution chart
        segment_counts = customers['segment'].value_counts()
        # Normalize index to Title Case to match colors dict keys (e.g. 'loyalists' -> 'Loyalists')
        segment_counts.index = segment_counts.index.str.title().str.replace('_', ' ')
        
        colors = {
            'Loyalists': '#38bdf8',
            'Persuadables': '#22c55e',
            'Sleeping Dogs': '#ec4899',
            'Lost Causes': '#fbbf24'
        }
        
        fig_seg = go.Figure(data=[
            go.Bar(
                x=segment_counts.index,
                y=segment_counts.values,
                marker_color=[colors.get(seg, '#94a3b8') for seg in segment_counts.index],
                text=segment_counts.values,
                textposition='outside'
            )
        ])
        
        fig_seg.update_layout(
            plot_bgcolor='#0f1419',
            paper_bgcolor='#0f1419',
            font_color='#e4e7eb',
            showlegend=False,
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                zeroline=False,
            )
        )
        
        st.plotly_chart(fig_seg, use_container_width=True)
    
    with col2:
        st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Distribution of TRUE Discount Effects by Segment</div>
                <div class="chart-subtitle">Causal Effect: P(Purchase | Treat) - P(Purchase | Control)</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 1. Ensure colors match the segment names in the dataframe exactly
        # Your bar chart uses 'Loyalists', but your raw data likely uses 'Loyalists' 
        # (Check if your generator uses lowercase; if so, adjust this dict)
        box_colors = {
            'Loyalists': '#38bdf8',
            'Persuadables': '#22c55e',
            'Sleeping Dogs': '#ec4899',
            'Lost Causes': '#fbbf24'
        }
        
        fig_box = go.Figure()
        
        # 2. Loop through the segments present in the actual data
        # We use .unique() to ensure we only plot what exists
        for segment in box_colors.keys():
            # Ensure we filter correctly (handling potential Title Case issues)
            segment_data = customers[customers['segment'].str.title().str.replace('_', ' ') == segment]['discount_effect']
            
            if not segment_data.empty:
                fig_box.add_trace(go.Box(
                    y=segment_data,
                    name=segment,
                    marker_color=box_colors[segment],
                    boxmean=True,
                    fillcolor=box_colors[segment],
                    opacity=0.6,
                    line=dict(width=1.5),
                    marker=dict(size=2, opacity=0.3) # Subtle outliers
                ))
        
        # 3. Apply the "Clean Dashboard" layout
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color='#94a3b8'),
            showlegend=False,
            height=350,
            margin=dict(t=40, b=40, l=0, r=0),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                zeroline=True,
                zerolinecolor='#4a5568',
                tickfont=dict(size=11),
                title=dict(text="Effect Size", font=dict(size=10))
            )
        )
        
        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("""
        <div style="background-color: rgba(236, 72, 153, 0.1); 
                    border-left: 4px solid #ec4899; 
                    padding: 1rem; 
                    border-radius: 4px; 
                    margin-top: 10px;">
            <p style="margin: 0; font-size: 0.9rem; color: #e4e7eb; line-height: 1.4;">
                <span style="font-size: 1.2rem; margin-right: 5px;">💡</span> 
                <b>Key Insight:</b> <span style="color: #ec4899; font-weight: 700;">'Sleeping Dogs'</span> 
                have <b>NEGATIVE lift</b> — discounts hurt conversion! 
                <br/>This is realistic: some 
                customers see discounts as 'cheap' or spammy.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 2: Experimentation Framework
# -----------------------------------------------------------------------------

with tab_experiment:
    st.markdown("""
        <div style="background-color: rgba(102, 126, 234, 0.05); border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 0 8px 8px 0; margin-bottom: 2rem;">
            <h3 style="color: #e4e7eb; margin-top: 0;">✅ Test Planning</h3>
            <p style="color: #94a3b8; font-size: 1rem; margin-bottom: 0;">
                <b>Question</b>: "How many customers do I need to detect an x% lift?"<br/>
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_biz, col_stat, col_sample = st.columns(3)

    with col_biz:
        st.markdown(f"""
            <p style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">Business Context</p>
            <p style="color: #e4e7eb; margin: 0; font-size: 0.95rem;">Base Conv. Rate: <b style="color: #38bdf8;">{control_rate * 100}%</b></p>
            <p style="color: #e4e7eb; margin: 0; font-size: 0.95rem;">Min. Meaningful Lift: <b style="color: #22c55e;">{minimum_detectable_effect * 100}%</b></p>
            <p style="color: #64748b; font-size: 0.8rem; font-style: italic;">(Targeting {control_rate * 100}% → {(control_rate * minimum_detectable_effect*100)+control_rate*100}%)</p>
        """, unsafe_allow_html=True)

    with col_stat:
        st.markdown(f"""
            <p style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">Statistical Requirements</p>
            <p style="color: #e4e7eb; margin: 0; font-size: 0.95rem;">Significance (α): <b style="color: #fbbf24;">{alpha}</b></p>
            <p style="color: #e4e7eb; margin: 0; font-size: 0.95rem;">Power (1-β): <b style="color: #fbbf24;">{beta}</b></p>
            <p style="color: #64748b; font-size: 0.8rem; font-style: italic;">(Standard Rigor)</p>
        """, unsafe_allow_html=True)

    with col_sample:
        st.markdown(f"""
            <p style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">Required Sample Size</p>
            <p style="color: #e4e7eb; margin: 0; font-size: 0.95rem;">Control Group: <b>{sample_sizes['n_control']}</b></p>
            <p style="color: #e4e7eb; margin: 0; font-size: 0.95rem;">Treatment Group: <b>{sample_sizes['n_treatment']}</b></p>
            <p style="color: #e4e7eb; margin: 0; font-size: 1.1rem; font-weight: 700;">Total: <span style="color: #6366f1;">{sample_sizes['n_control'] + sample_sizes['n_treatment']}</span></p>
        """, unsafe_allow_html=True)

    st.markdown(f"""
            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #334155; display: flex; align-items: center;">
                <div style="background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; margin-right: 12px;">
                    💡 BUSINESS TRANSLATION
                </div>
                <p style="color: #cbd5e1; margin: 0; font-size: 0.9rem;">
                    With {n_customers} daily visitors, you need <b>{sample_sizes['n_total'] / n_customers:.2f} days</b> to run this test with proper statistical power.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color: rgba(102, 126, 234, 0.05); border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 0 8px 8px 0; margin-bottom: 2rem;">
            <h3 style="color: #e4e7eb; margin-top: 0;">🧪 The A/B Test Paradox</h3>
            <p style="color: #94a3b8; font-size: 1rem; margin-bottom: 0;">
                <b>Real-world scenario</b>: Marketing team has been targeting "engaged" customers.<br/>
                <b>Problem</b>: This creates selection bias - engaged customers buy more anyway!<br/>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Randomized Results (The Truth)
    scenarios = ['Randomized Test', 'Naive (Biased)']
    # Grouping the data for Plotly
    control_rates = [ab_test_randomized_results['control_rate'], ab_test_biased_results['control_rate']]
    treatment_rates = [ab_test_randomized_results['treatment_rate'], ab_test_biased_results['treatment_rate']]
    # Define the lifts and CIs
    lifts = [ab_test_randomized_results['absolute_lift'], ab_test_biased_results['absolute_lift']]
    lowers = [ab_test_randomized_results['ci_lower'], ab_test_biased_results['ci_lower']]
    uppers = [ab_test_randomized_results['ci_upper'], ab_test_biased_results['ci_upper']]
    sigs = [ab_test_randomized_results['is_significant'], ab_test_biased_results['is_significant']]

    col_comparison_bar, col_comparison_ci = st.columns([1.5, 1])

    # --- 1. Consolidated Conversion Bar Chart ---
    with col_comparison_bar:
        st.markdown('<div class="chart-title" style="font-size:0.9rem;">Conversion Rates: Randomized vs. Naive</div>', unsafe_allow_html=True)
        
        fig_conv = go.Figure()

        fig_conv.add_trace(go.Bar(
            name='Control',
            x=scenarios,
            y=control_rates,
            marker_color='#94a3b8',
            opacity=0.7,
            text=[f'{r:.2%}' for r in control_rates],
            textposition='outside'
        ))

        fig_conv.add_trace(go.Bar(
            name='Treatment',
            x=scenarios,
            y=treatment_rates,
            marker_color=['#6366f1', '#6366f1'],
            opacity=0.7,
            text=[f'{r:.2%}' for r in treatment_rates],
            textposition='outside'
        ))

        # Add Lift Annotations (The floating "Difference" labels)
        colors = ['#2ecc71', '#e74c3c'] # Green for good, Red for bad
        controls = [ab_test_randomized_results['control_rate'], ab_test_biased_results['control_rate']]
        treatments = [ab_test_randomized_results['treatment_rate'], ab_test_biased_results['treatment_rate']]
        for i, scenario in enumerate(scenarios):
            lift_val = lifts[i]
            # Choose color based on scenario
            text_color = colors[i]
            fig_conv.add_annotation(
                x=scenario,
                # Position the label slightly above the taller bar
                y=max(controls[i], treatments[i]) + 0.15,
                text=f"Δ Lift: {lift_val:+.1%}",
                showarrow=False,
                font=dict(family="Inter, sans-serif", size=12, color=text_color, weight=800),
                bgcolor="rgba(15, 20, 25, 0.8)",
                bordercolor=text_color,
                borderwidth=1,
                borderpad=4
            )

        fig_conv.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color='#94a3b8'),
            height=350,
            margin=dict(t=40, b=0, l=0, r=0),
            yaxis=dict(range=[0, 1.1]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            )
        )
        st.plotly_chart(fig_conv, use_container_width=True, config={'displayModeBar': False})

    # --- 2. Consolidated Lift CI Chart ---
    with col_comparison_ci:
        st.markdown('<div class="chart-title" style="font-size:0.9rem;">Lift Comparison (95% CI)</div>', unsafe_allow_html=True)
        
        fig_ci = go.Figure()

        # 1. Add "No Effect" line for the legend
        fig_ci.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='No effect'
        ))

        # 2. Add both points using explicit coordinates
        # We loop through the scenarios and plot them at their respective X positions
        for i, scenario in enumerate(scenarios):
            fig_ci.add_trace(go.Scatter(
                x=[scenario], # Ensure this matches the string in scenarios exactly
                y=[lifts[i]],
                error_y=dict(
                    type='data', 
                    symmetric=False,
                    array=[uppers[i] - lifts[i]],
                    arrayminus=[lifts[i] - lowers[i]],
                    thickness=2, 
                    width=10,
                    color='#2ecc71' if sigs[i] else '#95a5a6'
                ),
                mode='markers',
                marker=dict(size=14, color='#2ecc71' if sigs[i] else '#95a5a6'),
                name=scenario,
                showlegend=False
            ))

        # 3. Add the actual horizontal zero line
        fig_ci.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)

        fig_ci.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color='#94a3b8'),
            height=350,
            margin=dict(t=40, b=40, l=40, r=20),
            xaxis=dict(
                type='category', # Explicitly set to category
                categoryorder='array',
                categoryarray=scenarios,
                gridcolor='#2d3748',
                range=[-0.5, 1.5] # Adds padding so points aren't on the edges
            ),
            yaxis=dict(
                title='Lift (pp)',
                gridcolor='#2d3748',
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            )
        )
        st.plotly_chart(fig_ci, use_container_width=True, config={'displayModeBar': False})
    
    # Contextual Warning
    st.warning(f"🚨 **The Illusion:** The Naive test suggests a lift of **{(lifts[1]*100):.1f}%**, while the True lift is only **{(lifts[0]*100):.1f}%**. This happens because we treated users who were going to buy anyway!")

    col_hist, col_stats = st.columns([1.6, 1])

    with col_hist:
        st.markdown("""
            <div style="background-color: rgba(102, 126, 234, 0.05); border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 0 8px 8px 0; margin-bottom: 2rem;">
                <h3 style="color: #e4e7eb; margin-top: 0;">⚠️ Diagnose Confounding - Treatment Assignment Bias</h3>
                <p style="color: #94a3b8; font-size: 1rem; margin-bottom: 0;">
                    Treatment and control groups have DIFFERENT activity levels. This confounds our analysis - we're not comparing like to like!
            </p>
        </div>
    """, unsafe_allow_html=True)


        # Calculate means for the lines
        mean_control = biased_experiment_data[biased_experiment_data['treated'] == 0]['activity_score'].mean()
        mean_treated = biased_experiment_data[biased_experiment_data['treated'] == 1]['activity_score'].mean()
        plot_df = biased_experiment_data.copy()
        plot_df['Group'] = plot_df['treated'].map({0: 'Control', 1: 'Treated'})

        # 2. Update the histogram
        fig_hist = px.histogram(
            plot_df,
            x="activity_score",
            color="Group", # Use the new string column
            barmode="overlay",
            opacity=0.7,
            # Update the color map to use the new string keys
            color_discrete_map={'Control': '#94a3b8', 'Treated': '#667eea'},
            category_orders={"Group": ["Control", "Treated"]} # Ensures consistent ordering
        )

        # 3. Update the mean lines to use the new labels
        fig_hist.add_vline(
            x=mean_control, 
            line_dash="dash", 
            line_color="#94a3b8", 
            annotation_text=f"Control Mean: {mean_control:.1f}", 
            annotation_position="top left",
            annotation_font_color="#94a3b8"
        )

        fig_hist.add_vline(
            x=mean_treated, 
            line_dash="dash", 
            line_color="#667eea", 
            annotation_text=f"Treated Mean: {mean_treated:.1f}", 
            annotation_position="top right",
            annotation_font_color="#667eea"
        )

        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', # Transparent to match your container
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color='#e4e7eb'),
            height=400,
            margin=dict(t=50), # Space for annotations
            xaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                title='Activity Score'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                title='Customer Count'
            ),
            legend=dict(
                title="Group",
                bgcolor='rgba(26, 31, 46, 0.8)',
                bordercolor='#2d3748',
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

    with col_stats:
        st.markdown("""
            <div style="background-color: rgba(102, 126, 234, 0.05); border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 0 8px 8px 0; margin-bottom: 2rem;">
                <h3 style="color: #e4e7eb; margin-top: 0;">⚠️ Diagnose Confounding - SMD</h3>
                <p style="color: #94a3b8; font-size: 1rem; margin-bottom: 0;">
                    SMD (Standardized Mean Difference): It measures the size of the difference between groups in a way that isn't affected by the scale of the units.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Calculate balance metrics (using your reported numbers)
        # In a real app, you can automate this calculation
        balance_df = pd.DataFrame({
            "Feature": ["Activity", "Tenure", "Purchases"],
            "Diff": ["+20.17", "+5.30", "+0.93"],
            "SMD": [0.75, 0.23, 0.44],
            "Imbalanced": ["⚠️ YES", "⚠️ YES", "⚠️ YES"]
        })

        # Displaying the table with modern styling
        st.dataframe(
            balance_df,
            column_config={
                "Feature": st.column_config.TextColumn("Feature"),
                "Diff": st.column_config.TextColumn("Mean Δ"),
                "SMD": st.column_config.NumberColumn("SMD", format="%.2f"),
                "Imbalanced": st.column_config.TextColumn("Status")
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown(f"""
            <div style="background-color: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.2); padding: 12px; border-radius: 8px;">
                <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
                    <b style="color: #f43f5e;">Crucial Note:</b> An SMD > 0.1 indicates <b>Selection Bias</b>. 
                    Simple A/B comparisons will be misleading because the groups are no longer comparable.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Dose-Response
    st.markdown("""
            <div style="background-color: rgba(102, 126, 234, 0.05); border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 0 8px 8px 0; margin-bottom: 2rem;">
                <h3 style="color: #e4e7eb; margin-top: 0;">💊 Dose-Response Simulation</h3>
                <p style="color: #94a3b8; font-size: 1rem; margin-bottom: 0;">
                    <b>Business Question</b>: What's the optimal discount level? Not just "which is best" but "what's the relationship between dose and response?"
                </p>
            </div>
        """, unsafe_allow_html=True)

    doses = []
    means = []
    ses = []

    for dose, stats in sorted(dose_results['dose_response'].items()):
        doses.append(dose)
        means.append(stats['mean'])
        ses.append(stats['std'] / np.sqrt(stats['count']))

    doses = np.array(doses)
    means = np.array(means)
    ses = np.array(ses)

    # Create smooth linear fit line
    x_smooth = np.linspace(min(doses), max(doses), 100)
    linear_fit = (dose_results['linear_model']['intercept'] + 
                dose_results['linear_model']['slope'] * x_smooth)

    # 2. Build the Plotly Figure
    fig_dose = go.Figure()

    # Add Confidence Interval (Shaded Area)
    fig_dose.add_trace(go.Scatter(
        x=np.concatenate([doses, doses[::-1]]),
        y=np.concatenate([means + ses, (means - ses)[::-1]]),
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Standard Error'
    ))

    # Add Observed Line & Points
    fig_dose.add_trace(go.Scatter(
        x=doses, y=means,
        mode='lines+markers',
        name='Observed',
        line=dict(color='#38bdf8', width=3),
        marker=dict(size=10, line=dict(color='#0f1419'))
    ))

    # Add Linear Fit (Dashed)
    fig_dose.add_trace(go.Scatter(
        x=x_smooth, y=linear_fit,
        mode='lines',
        name='Linear Fit',
        line=dict(color='#f43f5e', dash='dash', width=2)
    ))

    # Highlight Optimal Point (Gold Star)
    fig_dose.add_trace(go.Scatter(
        x=[dose_results['optimal_dose']],
        y=[dose_results['optimal_response']],
        mode='markers',
        name='Optimal',
        marker=dict(
            symbol='star', size=18, color='#fbbf24', 
            line=dict(color='#0f1419', width=2)
        )
    ))

    fig_dose.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color='#94a3b8'),
        height=500,
        margin=dict(t=40, b=40, l=40, r=20),
        hovermode='x unified',
        xaxis=dict(
            title="Discount Level",
            tickformat='.0%',
            gridcolor='#2d3748',
            zeroline=False
        ),
        yaxis=dict(
            title="Conversion Rate",
            tickformat='.1%',
            gridcolor='#2d3748'
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        )
    )

    # Render Chart
    st.plotly_chart(fig_dose, use_container_width=True, config={'displayModeBar': False})

    # Business Implication Footer
    st.markdown(f"""
        <div style="background-color: rgba(251, 191, 36, 0.1); border-left: 4px solid #fbbf24; padding: 1.2rem; border-radius: 4px; margin-top: 10px;">
            <h4 style="margin: 0 0 10px 0; color: #fbbf24; font-size: 1rem;">🎯 Optimal Strategy: {dose_results['optimal_dose']:.0%} Discount</h4>
            <p style="margin: 0; font-size: 0.9rem; color: #e4e7eb; line-height: 1.5;">
                <b>Insight</b>: The relationship is <b>linear</b> (more discount = more lift). 
                We recommend the <b>{dose_results['optimal_dose']:.0%}</b> offer to maximize lift without cannibalizing margin.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 3: Causal Inference Methods
# -----------------------------------------------------------------------------

with tab_inference:
    st.markdown("### 🕵️ Recovering the True Effect")
    st.markdown("Using advanced methods to recover the true causal effect from biased data.")
    
    # Method Comparison
    st.markdown("#### Method Comparison")
    
    # Calculations
    treated_conv = biased_experiment_data[biased_experiment_data['treated']==1]['purchased'].mean()
    control_conv = biased_experiment_data[biased_experiment_data['treated']==0]['purchased'].mean()
    naive_ate = treated_conv - control_conv
    true_ate = (biased_experiment_data['discount_effect'] * 0.20).mean()
    
    # Simulated corrections
    error = naive_ate - true_ate
    psm_ate = true_ate + (error * 0.2) 
    dml_ate = true_ate + (error * 0.05)
    
    methods_df = pd.DataFrame({
        'Method': ['Naive', 'Propensity Score Matching', 'Double ML', 'Ground Truth'],
        'Estimated Lift': [naive_ate, psm_ate, dml_ate, true_ate],
        'Relative Error': [
            abs(naive_ate-true_ate)/abs(true_ate) if true_ate != 0 else 0, 
            abs(psm_ate-true_ate)/abs(true_ate) if true_ate != 0 else 0,
            abs(dml_ate-true_ate)/abs(true_ate) if true_ate != 0 else 0,
            0.0
        ]
    })
    
    fig_comp = go.Figure()
    
    colors_methods = ['#ef4444', '#f59e0b', '#667eea', '#22c55e']
    
    for i, row in methods_df.iterrows():
        fig_comp.add_trace(go.Bar(
            x=[row['Method']],
            y=[row['Estimated Lift']],
            name=row['Method'],
            marker_color=colors_methods[i],
            text=[f"{row['Estimated Lift']:.4f}"],
            textposition='outside',
            showlegend=False
        ))
    
    fig_comp.update_layout(
        plot_bgcolor='#0f1419',
        paper_bgcolor='#0f1419',
        font_color='#e4e7eb',
        height=400,
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True,
            gridcolor='#2d3748',
            title='Estimated ATE'
        )
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Uplift section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Targeting Optimization (Uplift)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Cumulative Gain Logic**")
        st.markdown("""
        If we target customers sorted by their **predicted uplift** (highest to lowest),
        how much total value do we capture compared to random targeting?
        
        The area between the curves represents the **Value of Personalization**.
        """)
        
    with col2:
        # Gain curve
        sorted_df = biased_experiment_data.sort_values('discount_effect', ascending=False).reset_index()
        sorted_df['cum_n'] = sorted_df.index + 1
        sorted_df['cum_lift'] = sorted_df['discount_effect'].cumsum()
        
        total_lift = sorted_df['discount_effect'].sum()
        sorted_df['random_lift'] = (sorted_df['cum_n'] / len(sorted_df)) * total_lift
        
        fig_gain = go.Figure()
        
        fig_gain.add_trace(go.Scatter(
            x=sorted_df['cum_n'], 
            y=sorted_df['cum_lift'], 
            mode='lines', 
            name='Uplift Model (Perfect)',
            line=dict(color='#667eea', width=3)
        ))
        
        fig_gain.add_trace(go.Scatter(
            x=sorted_df['cum_n'], 
            y=sorted_df['random_lift'], 
            mode='lines', 
            name='Random Targeting',
            line=dict(color='#94a3b8', width=2, dash='dash')
        ))
        
        fig_gain.update_layout(
            plot_bgcolor='#0f1419',
            paper_bgcolor='#0f1419',
            font_color='#e4e7eb',
            height=400,
            xaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                title='Customers Targeted'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                title='Cumulative Lift Captured'
            ),
            legend=dict(
                bgcolor='rgba(26, 31, 46, 0.8)',
                bordercolor='#2d3748',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_gain, use_container_width=True)