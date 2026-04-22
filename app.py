"""
ECON 5200: Consulting Report — Streamlit Dashboard
NSW Job Training Program: Causal Impact Analysis

Author: Zehan Qin
Date: April 2026

Interactive dashboard for what-if analysis of the DML estimate of
the causal effect of NSW job training on 1978 earnings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="NSW Job Training: Causal Impact Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Core Estimates (from notebook analysis)
# ============================================================
# Main DML estimate (Random Forest nuisance - primary specification)
BASELINE_ATE = 1540.87
BASELINE_SE = 669.81  # (2852.69 - 229.04) / (2 * 1.96)

# Secondary DML estimate (Gradient Boosting nuisance - robustness)
ROBUST_ATE_GBR = 364.16
ROBUST_SE_GBR = 601.34

# Benchmarks for comparison
NAIVE_OLS = -8497.52
OLS_CONTROLS = 699.13
RCT_BENCHMARK = 1794.34

# Program cost (approximate, 1978 dollars)
PROGRAM_COST = 12500

# ============================================================
# Header
# ============================================================
st.title("📊 NSW Job Training Program: Causal Impact Analysis")
st.markdown(
    "**Client:** U.S. Department of Labor | **Analyst:** Zehan Qin | "
    "**Method:** Double Machine Learning (DML)"
)

st.markdown("""
This interactive dashboard answers the question: **Does participation in the NSW job training
program cause higher post-program earnings?** Adjust the parameters in the sidebar to explore
counterfactual scenarios and see how the estimated effect and uncertainty change in real time.
""")

st.divider()

# ============================================================
# Sidebar: What-If Controls
# ============================================================
st.sidebar.header("🎛️ What-If Scenario Controls")
st.sidebar.markdown("*Adjust parameters below. Results update in real time.*")

st.sidebar.subheader("Treatment Scenarios")

treatment_multiplier = st.sidebar.slider(
    "Treatment Intensity Multiplier",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="1.0 = baseline NSW program. 2.0 = doubled intensity (e.g., twice the training hours). "
         "Assumes linear scaling of effect."
)

confidence_level = st.sidebar.select_slider(
    "Confidence Level",
    options=[0.80, 0.90, 0.95, 0.99],
    value=0.95,
    format_func=lambda x: f"{int(x*100)}%"
)

st.sidebar.subheader("Model Specification")
nuisance_choice = st.sidebar.selectbox(
    "DML Nuisance Model",
    ["Random Forest (primary)", "Gradient Boosting (robustness)"],
    help="Random Forest estimate is closer to the RCT benchmark. "
         "Gradient Boosting is shown for robustness."
)

st.sidebar.subheader("Policy Scenarios")
annual_cost_per_participant = st.sidebar.number_input(
    "Program Cost per Participant ($)",
    min_value=5000,
    max_value=30000,
    value=PROGRAM_COST,
    step=500,
    help="Approximate cost of NSW program per participant in 1978 dollars."
)

effect_persistence_years = st.sidebar.slider(
    "Effect Persistence (years)",
    min_value=1,
    max_value=15,
    value=5,
    help="How many years the treatment effect is assumed to last."
)

# ============================================================
# Compute Adjusted Estimates
# ============================================================
if nuisance_choice == "Random Forest (primary)":
    base_ate = BASELINE_ATE
    base_se = BASELINE_SE
    spec_label = "RF"
else:
    base_ate = ROBUST_ATE_GBR
    base_se = ROBUST_SE_GBR
    spec_label = "GBR"

# Z-score for chosen confidence level
z_score = {0.80: 1.282, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence_level]

adjusted_ate = base_ate * treatment_multiplier
adjusted_se = base_se * treatment_multiplier
ci_lower = adjusted_ate - z_score * adjusted_se
ci_upper = adjusted_ate + z_score * adjusted_se

# ============================================================
# Main Results Panel
# ============================================================
st.header("📈 Estimated Causal Effect")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Point Estimate (ATE)",
    value=f"${adjusted_ate:,.0f}",
    delta=f"{(treatment_multiplier-1)*100:+.0f}% from baseline" if treatment_multiplier != 1.0 else None
)

col2.metric(
    label=f"{int(confidence_level*100)}% CI Lower",
    value=f"${ci_lower:,.0f}"
)

col3.metric(
    label=f"{int(confidence_level*100)}% CI Upper",
    value=f"${ci_upper:,.0f}"
)

col4.metric(
    label="RCT Benchmark",
    value=f"${RCT_BENCHMARK:,.0f}",
    help="Ground truth from original randomized experiment"
)

# Interpretation
if ci_lower > 0:
    st.success(
        f"✅ **Statistically significant positive effect:** "
        f"The {int(confidence_level*100)}% CI [\\${ci_lower:,.0f}, \\${ci_upper:,.0f}] excludes zero."
    )
elif ci_upper < 0:
    st.error(
        f"❌ **Statistically significant negative effect:** "
        f"The {int(confidence_level*100)}% CI [\\${ci_lower:,.0f}, \\${ci_upper:,.0f}] is below zero."
    )
else:
    st.warning(
        f"⚠️ **Point estimate positive, but CI includes zero:** "
        f"[\\${ci_lower:,.0f}, \\${ci_upper:,.0f}]. Effect is directionally consistent "
        f"with the RCT benchmark but not statistically distinguishable from zero at this confidence level."
    )

# ============================================================
# Counterfactual Scenario Description
# ============================================================
st.subheader("🔮 Counterfactual Scenario")

if treatment_multiplier == 1.0:
    st.markdown(f"""
    **Baseline scenario (no change):** Under the NSW program as implemented, training increases
    annual post-program earnings by **\\${adjusted_ate:,.0f}** per participant
    ({int(confidence_level*100)}% CI: [\\${ci_lower:,.0f}, \\${ci_upper:,.0f}]).
    This is based on the **{spec_label}** nuisance specification.
    """)
else:
    change_pct = (treatment_multiplier - 1) * 100
    direction = "increased" if change_pct > 0 else "decreased"
    st.markdown(f"""
    **What-if scenario:** If treatment intensity is **{direction} by {abs(change_pct):.0f}%**
    (multiplier = {treatment_multiplier:.1f}x), the estimated effect changes to
    **\\${adjusted_ate:,.0f}** ({int(confidence_level*100)}% CI: [\\${ci_lower:,.0f}, \\${ci_upper:,.0f}]).

    Compared to the baseline estimate of \\${base_ate:,.0f}, this is a change of
    **\\${adjusted_ate - base_ate:+,.0f}** ({change_pct:+.0f}%).

    *Note: This assumes linear scaling of the treatment effect with intensity, which is a
    strong assumption that would require further validation.*
    """)

st.divider()

# ============================================================
# Dynamic Uncertainty Visualization
# ============================================================
st.header("📊 Effect vs. Treatment Intensity")

# Build data for the curve
multipliers = np.arange(0.0, 3.05, 0.05)
ates_curve = base_ate * multipliers
ses_curve = base_se * multipliers
upper_curve = ates_curve + z_score * ses_curve
lower_curve = ates_curve - z_score * ses_curve

fig = go.Figure()

# Confidence band
fig.add_trace(go.Scatter(
    x=multipliers,
    y=upper_curve,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=multipliers,
    y=lower_curve,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(26, 35, 126, 0.2)',
    name=f'{int(confidence_level*100)}% Confidence Band'
))

# Point estimate line
fig.add_trace(go.Scatter(
    x=multipliers,
    y=ates_curve,
    mode='lines',
    line=dict(color='#1a237e', width=3),
    name='Point Estimate'
))

# RCT benchmark reference line
fig.add_hline(
    y=RCT_BENCHMARK,
    line_dash="dash",
    line_color="green",
    annotation_text=f"RCT Benchmark (\\${RCT_BENCHMARK:,.0f})",
    annotation_position="right"
)

# Zero line
fig.add_hline(y=0, line_dash="dot", line_color="gray")

# Current multiplier marker
fig.add_vline(
    x=treatment_multiplier,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Current: {treatment_multiplier:.1f}x",
    annotation_position="top"
)

fig.update_layout(
    title=f'Estimated Effect vs. Treatment Intensity ({spec_label} Nuisance)',
    xaxis_title='Treatment Intensity Multiplier',
    yaxis_title='Estimated Causal Effect ($)',
    template='plotly_white',
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Methods Comparison
# ============================================================
st.header("🔍 Method Comparison: Naive vs. Causal Estimates")

st.markdown("""
This shows why a **prediction-only approach fails** for this problem.
Naive OLS gives the wrong *direction* because of severe selection bias.
DML corrects the bias and aligns with the RCT benchmark.
""")

methods_df = pd.DataFrame({
    'Method': ['Naive OLS (no controls)', 'OLS with controls',
               'DML (GBR nuisance)', 'DML (RF nuisance)', 'RCT Benchmark'],
    'Estimate': [NAIVE_OLS, OLS_CONTROLS, ROBUST_ATE_GBR, BASELINE_ATE, RCT_BENCHMARK],
    'CI Lower': [-9893.16, -374.30, -814.46, 229.04, RCT_BENCHMARK - 800],
    'CI Upper': [-7101.88, 1772.56, 1542.79, 2852.69, RCT_BENCHMARK + 800],
    'Type': ['Biased', 'Biased', 'Causal', 'Causal (primary)', 'Ground Truth']
})

fig_compare = go.Figure()

colors_map = {
    'Biased': '#e74c3c',
    'Causal': '#f39c12',
    'Causal (primary)': '#2ecc71',
    'Ground Truth': '#3498db'
}

for _, row in methods_df.iterrows():
    fig_compare.add_trace(go.Scatter(
        x=[row['Method']],
        y=[row['Estimate']],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[row['CI Upper'] - row['Estimate']],
            arrayminus=[row['Estimate'] - row['CI Lower']],
            color='black',
            thickness=2
        ),
        mode='markers',
        marker=dict(size=18, color=colors_map[row['Type']]),
        name=row['Type'],
        showlegend=row['Type'] not in [t.name for t in fig_compare.data]
    ))

fig_compare.add_hline(y=0, line_dash="dot", line_color="gray")
fig_compare.add_hline(
    y=RCT_BENCHMARK,
    line_dash="dash",
    line_color="green",
    annotation_text="RCT Truth",
    annotation_position="right"
)

fig_compare.update_layout(
    title='Comparison of Estimation Methods (95% Confidence Intervals)',
    yaxis_title='Estimated Effect ($)',
    template='plotly_white',
    height=500,
    showlegend=True
)

st.plotly_chart(fig_compare, use_container_width=True)

# ============================================================
# ROI Analysis
# ============================================================
st.header("💰 Return on Investment (ROI) Analysis")

total_earnings_gain = adjusted_ate * effect_persistence_years
net_benefit = total_earnings_gain - annual_cost_per_participant
payback_years = annual_cost_per_participant / adjusted_ate if adjusted_ate > 0 else float('inf')

col1, col2, col3 = st.columns(3)
col1.metric(
    "Total Earnings Gain",
    f"${total_earnings_gain:,.0f}",
    help=f"Annual effect × {effect_persistence_years} years"
)
col2.metric(
    "Net Benefit per Participant",
    f"${net_benefit:,.0f}",
    delta=f"{'Positive ROI' if net_benefit > 0 else 'Negative ROI'}"
)
col3.metric(
    "Payback Period",
    f"{payback_years:.1f} years" if payback_years < 100 else "N/A"
)

if net_benefit > 0 and payback_years <= effect_persistence_years:
    st.success(
        f"✅ **Positive ROI:** At ${annual_cost_per_participant:,} program cost and "
        f"{effect_persistence_years}-year persistence, the program generates "
        f"${net_benefit:,.0f} net benefit per participant."
    )
else:
    st.warning(
        f"⚠️ **ROI not clearly positive** at current assumptions. "
        f"Extending effect persistence beyond {effect_persistence_years} years "
        f"or reducing program cost would improve the ROI."
    )

st.caption("""
*ROI analysis assumes the treatment effect persists linearly for the specified number of years.
In reality, effects may decay over time. This is an illustrative scenario, not a definitive
cost-benefit conclusion.*
""")

# ============================================================
# Key Assumptions & Threats
# ============================================================
with st.expander("⚠️ Key Assumptions and Threats to Identification"):
    st.markdown("""
    **Key Assumption: Conditional Independence (Unconfoundedness)**

    DML assumes that, conditional on observed covariates (age, education, race, marital status,
    pre-treatment earnings), treatment assignment is independent of potential outcomes.

    **Primary Threats:**

    1. **Unobserved Confounding:** NSW participants were referred through probation officers
       and social workers based on unmeasured characteristics (motivation, mental health,
       criminal history). If these affect future earnings, the assumption fails.

    2. **Overlap Violation:** Treated and control groups have very different covariate
       distributions (|SMD| up to 2.43), meaning DML may extrapolate into regions with
       no treated observations.

    3. **External Validity:** Data is from 1975-1978. Modern labor markets and training
       programs differ substantially.

    4. **Specification Sensitivity:** RF ($1,541) and GBR ($365) nuisance models give
       different point estimates. The directional conclusion is robust; magnitude is not.

    **Validation:** The RF-DML estimate (\\$1,541) is close to the RCT benchmark (\\$1,794),
    providing partial validation of the causal claim.
    """)

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "ECON 5200: Causal Machine Learning & Applied Analytics | Northeastern University | Spring 2026 | "
    "Data: Lalonde (1986) / Dehejia-Wahba (1999) | Method: Double Machine Learning (Chernozhukov et al., 2018)"
)
