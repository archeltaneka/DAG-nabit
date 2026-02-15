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
def load_data(n_customers, random_seed, confounding_level, include_noise, rare_events):
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
    
    # 2. Simulate Biased Experiment
    simulator = BehaviorSimulator(customers, config)
    
    # Determine bias method based on level
    if confounding_level == "None (Randomized Trial)":
        assignment = 'random'
    elif confounding_level == "Moderate (Activity Bias)":
        assignment = 'biased_activity'
    else: # "High (Value Bias)"
        assignment = 'biased_value'
        
    experiment_data = simulator.simulate_experiment(
        treatment_assignment=assignment,
        discount_amount=0.20
    )
    
    return customers, experiment_data

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
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
    
    st.number_input(
        "Simulation Seed",
        value=42,
        min_value=1,
        key="random_seed",
        help="Fixed seed ensures reproducibility of the synthetic dataset"
    )
    
    st.markdown("---")
    
    # Toggles
    include_noise = st.toggle("Include Noise", value=True, help="Add random noise to activity scores")
    rare_events = st.toggle("Rare Events", value=False, help="Include rare customer behaviors")
    
    st.markdown("---")
    
    # Generate button
    if st.button("⚡ Generate Population", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Last generated timestamp
    st.caption("Last generated: 2 mins ago")

# Get simulation parameters
random_seed = st.session_state.get("random_seed", 42)
confounding_level = "Moderate (Activity Bias)"  # Can be made dynamic

# Load Data
customers, df = load_data(n_customers, random_seed, confounding_level, include_noise, rare_events)

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
    "Experimentation Framework", 
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
    
    # Updated Scatter Plot with Synced Colors
    st.markdown("""
        <div class="chart-container">
            <div class="chart-title">True Causal Effects vs. Observable Features</div>
            <div class="chart-subtitle">Reveals the correlation structure between treatment effects and customer activity</div>
        </div>
    """, unsafe_allow_html=True)

    # Use the same color map we established for the rest of the UI
    colors_map = {
        'Loyalists': '#38bdf8',
        'Persuadables': '#22c55e',
        'Sleeping Dogs': '#ec4899',
        'Lost Causes': '#fbbf24'
    }

    # Create a display-ready column for the legend
    plot_df_scatter = customers.sample(min(2000, len(customers))).copy()
    plot_df_scatter['Segment'] = plot_df_scatter['segment'].str.title().str.replace('_', ' ')

    fig_scatter = px.scatter(
        plot_df_scatter,
        x="activity_score",
        y="discount_effect",
        color="Segment",
        size="account_value",
        hover_data=["age", "tenure_months"],
        color_discrete_map=colors_map, # This syncs the colors
        labels={
            "discount_effect": "True Lift in Purchase Prob (pp)", 
            "activity_score": "Activity Score (0-100)"
        },
        category_orders={"Segment": ["Loyalists", "Persuadables", "Sleeping Dogs", "Lost Causes"]}
    )

    fig_scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color='#94a3b8'),
        height=500,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='#2d3748',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#2d3748',
            zeroline=True,
            zerolinecolor='#4a5568'
        ),
        legend=dict(
            title="",
            bgcolor='rgba(26, 31, 46, 0.8)',
            bordercolor='#2d3748',
            borderwidth=1,
            font=dict(size=11, color='#e4e7eb'),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Refine marker appearance
    fig_scatter.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='#0f1419')))

    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

# -----------------------------------------------------------------------------
# Tab 2: Experimentation Framework
# -----------------------------------------------------------------------------

with tab_experiment:
    st.markdown("### 🧪 The Observable Bias")
    st.markdown("This is what we see in the **real world**: observing outcomes without knowing counterfactuals.")
    
    # Check for bias
    is_biased = confounding_level != "None (Randomized Trial)"
    
    if is_biased:
        st.warning(f"⚠️ **Selection Bias Detected**: {confounding_level}. Treated group is NOT comparable to Control.")
    else:
        st.success("✅ **RCT**: Random Assignment. Treated and Control groups are comparable.")

    # Row 1: Bias Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Treatment Assignment Bias</div>
            </div>
        """, unsafe_allow_html=True)
        
        fig_hist = px.histogram(
            df,
            x="activity_score",
            color="treated",
            barmode="overlay",
            labels={"treated": "Treated"},
            opacity=0.7,
            color_discrete_map={0: '#94a3b8', 1: '#667eea'}
        )
        
        fig_hist.update_layout(
            plot_bgcolor='#0f1419',
            paper_bgcolor='#0f1419',
            font_color='#e4e7eb',
            height=400,
            xaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                title='Activity Score'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
            ),
            legend=dict(
                bgcolor='rgba(26, 31, 46, 0.8)',
                bordercolor='#2d3748',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Naive Estimates (Misleading!)</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Naive ATE calculation
        treated_conv = df[df['treated']==1]['purchased'].mean()
        control_conv = df[df['treated']==0]['purchased'].mean()
        naive_lift = treated_conv - control_conv
        
        # True ATE calculation
        true_ate = df['discount_effect'].mean() * 0.20
        
        fig_bar = go.Figure(data=[
            go.Bar(
                name='Naive Look', 
                x=['Lift'], 
                y=[naive_lift], 
                marker_color='#ef4444' if is_biased else '#667eea',
                text=[f'{naive_lift:.4f}'],
                textposition='outside'
            ),
            go.Bar(
                name='True Causal Effect', 
                x=['Lift'], 
                y=[true_ate], 
                marker_color='#22c55e',
                text=[f'{true_ate:.4f}'],
                textposition='outside'
            )
        ])
        
        fig_bar.update_layout(
            plot_bgcolor='#0f1419',
            paper_bgcolor='#0f1419',
            font_color='#e4e7eb',
            barmode='group',
            height=400,
            showlegend=True,
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
            ),
            legend=dict(
                bgcolor='rgba(26, 31, 46, 0.8)',
                bordercolor='#2d3748',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.metric(
            label="Estimation Error", 
            value=f"{abs(naive_lift - true_ate):.4f}", 
            delta="Lower is better", 
            delta_color="inverse"
        )

    # Dose-Response
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💊 Dose-Response Simulation")
    st.markdown("What happens if we vary the discount amount?")
    
    if st.button("Run Dose-Response Simulation"):
        doses = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        results = []
        
        config_temp = SimulationConfig(n_customers=1000, random_seed=random_seed)
        cust_temp = generate_customer_data(config_temp)
        sim_temp = BehaviorSimulator(cust_temp, config_temp)
        
        for d in doses:
            d_df = sim_temp.simulate_experiment(treatment_assignment='random', discount_amount=d)
            conv_rate = d_df[d_df['treated']==1]['purchased'].mean()
            results.append({'Discount': d, 'Conversion': conv_rate})
            
        dr_df = pd.DataFrame(results)
        
        fig_dr = px.line(
            dr_df, 
            x="Discount", 
            y="Conversion", 
            markers=True,
        )
        
        fig_dr.update_traces(
            line_color='#667eea',
            marker=dict(size=10, color='#667eea')
        )
        
        fig_dr.update_layout(
            plot_bgcolor='#0f1419',
            paper_bgcolor='#0f1419',
            font_color='#e4e7eb',
            height=400,
            xaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                title='Discount Amount'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#2d3748',
                title='Conversion Rate'
            )
        )
        
        st.plotly_chart(fig_dr, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 3: Causal Inference Methods
# -----------------------------------------------------------------------------

with tab_inference:
    st.markdown("### 🕵️ Recovering the True Effect")
    st.markdown("Using advanced methods to recover the true causal effect from biased data.")
    
    # Method Comparison
    st.markdown("#### Method Comparison")
    
    # Calculations
    treated_conv = df[df['treated']==1]['purchased'].mean()
    control_conv = df[df['treated']==0]['purchased'].mean()
    naive_ate = treated_conv - control_conv
    true_ate = (df['discount_effect'] * 0.20).mean()
    
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
        sorted_df = df.sort_values('discount_effect', ascending=False).reset_index()
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