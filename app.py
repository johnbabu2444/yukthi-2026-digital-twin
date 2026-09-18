import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# =====================================================================
# 1. INDUSTRIAL MISSION CONTROL UI CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="YUKTHI 2026 - Digital Twin Carbon & Optimization Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #080c14; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .metric-card { background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; text-align: left; }
    .card-title { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
    .card-value { font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
    .status-green { color: #10b981; }
    .status-amber { color: #f59e0b; }
    .status-red { color: #ef4444; }
    .status-cyan { color: #06b6d4; }
    .directive-card { background-color: #0f172a; border-left: 4px solid #3b82f6; padding: 12px 16px; margin-bottom: 12px; border-radius: 0 6px 6px 0; }
    .risk-high { border-left-color: #ef4444 !important; }
    .risk-med { border-left-color: #f59e0b !important; }
    .risk-low { border-left-color: #10b981 !important; }
    .code-badge { background-color: #1e293b; color: #38bdf8; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Header Title
st.title("🛡️ DIGITAL TWIN PREDICTIVE MAINTENANCE & CARBON COMMAND CENTER")
st.caption("YUKTHI 2026 ENTERPRISE SUITE | REAL-TIME OPTIMIZATION & DEGRADATION FORECASTING")
st.divider()

# =====================================================================
# 2. STRICT CSV-ONLY DATA ENGINE
# =====================================================================

def load_data():
    try:
        df = pd.read_csv('anomalies_detected.csv')
    except Exception as e:
        st.error("⚠️ Primary data artifact `anomalies_detected.csv` not found. Please ensure `detect_anomalies.py` has been executed.")
        st.stop()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df_raw = load_data()

# =====================================================================
# 3. INTERACTIVE SIDEBAR CONTROLS
# =====================================================================

st.sidebar.markdown("### 🎛️ Digital Twin Controls")

# --- LIVE TELEMETRY STREAM TOGGLE ---
st.sidebar.markdown("### 📡 Live Telemetry Stream")
live_mode = st.sidebar.checkbox("Enable Live IoT Stream Auto-Refresh")

if live_mode:
    st.sidebar.caption("🟢 Live IoT Feed Active — Auto-Refreshing (3s)")
    time.sleep(3)
    st.rerun()

selected_chiller = st.sidebar.selectbox("Monitored Asset Unit", df_raw['equipment_id'].unique())

sensitivity_threshold = st.sidebar.slider(
    "ML Anomaly Threshold Cutoff (s)",
    min_value=0.40,
    max_value=0.75,
    value=0.54,
    step=0.01,
    help="Dynamically recalculates Isolation Forest flags, degradation vectors, and financial metrics."
)

min_date = df_raw['timestamp'].min().date()
max_date = df_raw['timestamp'].max().date()
date_range = st.sidebar.date_input("Telemetry Date Window", [min_date, max_date], min_value=min_date, max_value=max_date)

if st.sidebar.button("🚨 Inject Stress Test Anomaly"):
    st.sidebar.warning(f"Injected synthetic thermal overload into {selected_chiller}!")
    last_idx = df_raw[df_raw['equipment_id'] == selected_chiller].index[-1]
    df_raw.loc[last_idx, 'Chiller Energy Consumption (kWh)'] *= 1.85
    df_raw.loc[last_idx, 'anomaly_score'] = 0.94
    df_raw.loc[last_idx, 'driver_matrix'] = "Chiller Energy Consumption (kWh):70|Building Load (RT):30"

# =====================================================================
# 4. RE-EVALUATE DYNAMIC ML METRICS FROM CSV Telemetry
# =====================================================================

df_raw['is_anomaly'] = (df_raw['anomaly_score'] > sensitivity_threshold).astype(int)
df_raw['mechanical_integrity'] = 100.0 * (1.0 - df_raw['anomaly_score'])
df_raw['mechanical_integrity_3h'] = df_raw.groupby('equipment_id')['mechanical_integrity'].transform(lambda x: x.rolling(6, min_periods=1).mean())

if len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df_filtered = df_raw[(df_raw['timestamp'] >= start_dt) & (df_raw['timestamp'] < end_dt)].copy()
else:
    df_filtered = df_raw.copy()

active_df = df_filtered[df_filtered['equipment_id'] == selected_chiller].copy()

# =====================================================================
# 5. DYNAMIC CARBON & FINANCIAL METRICS (CSV-Sourced)
# =====================================================================

anom_df = active_df[active_df['is_anomaly'] == 1]

if not anom_df.empty:
    total_carbon_leakage_kg = anom_df['carbon_leakage_kg'].sum() if 'carbon_leakage_kg' in anom_df.columns else 0.0
    total_financial_loss_inr = anom_df['financial_leakage_inr'].sum() if 'financial_leakage_inr' in anom_df.columns else 0.0
    hours_span = max(1.0, len(active_df) * 0.5)
    financial_leakage_rate_inr = total_financial_loss_inr / hours_span
else:
    total_carbon_leakage_kg = 0.0
    total_financial_loss_inr = 0.0
    financial_leakage_rate_inr = 0.0

unit_integrity = active_df['mechanical_integrity_3h'].iloc[-1] if not active_df.empty else 100.0
unit_risk_vector = active_df['degradation_risk_vector'].iloc[-1] if ('degradation_risk_vector' in active_df.columns and not active_df.empty) else 0.0
grid_stability_idx = max(0.0, 100.0 - unit_risk_vector * 1.2)

def get_integrity_color_class(val):
    if val > 80: return "status-green"
    elif val >= 55: return "status-amber"
    else: return "status-red"

# =====================================================================
# 6. HEADER MISSION CONTROL INDICATORS
# =====================================================================

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    h_class = get_integrity_color_class(unit_integrity)
    st.markdown(f'''
    <div class="metric-card">
        <div class="card-title">Fleet Mechanical Integrity</div>
        <div class="card-value {h_class}">{unit_integrity:.1f}%</div>
    </div>''', unsafe_allow_html=True)

with m2:
    st.markdown(f'''
    <div class="metric-card">
        <div class="card-title">Active Leakage Rate</div>
        <div class="card-value status-amber">₹{financial_leakage_rate_inr:,.1f}/hr</div>
    </div>''', unsafe_allow_html=True)

with m3:
    st.markdown(f'''
    <div class="metric-card">
        <div class="card-title">Excess Carbon Footprint</div>
        <div class="card-value status-cyan">{total_carbon_leakage_kg:,.1f} kg</div>
    </div>''', unsafe_allow_html=True)

with m4:
    st.markdown(f'''
    <div class="metric-card">
        <div class="card-title">Grid Load Stability</div>
        <div class="card-value status-green">{grid_stability_idx:.1f}%</div>
    </div>''', unsafe_allow_html=True)

with m5:
    st.markdown(f'''
    <div class="metric-card">
        <div class="card-title">Degradation Risk Vector</div>
        <div class="card-value status-red">{unit_risk_vector:.1f}%</div>
    </div>''', unsafe_allow_html=True)

st.write("")

# Sidebar Deliverable Exports
st.sidebar.divider()
st.sidebar.markdown("### 📄 Executive Deliverables")
export_cols = [c for c in ['timestamp', 'equipment_id', 'Chiller Energy Consumption (kWh)', 'digital_twin_baseline_kwh', 'anomaly_score', 'carbon_leakage_kg', 'financial_leakage_inr', 'driver_matrix'] if c in active_df.columns]
export_df = active_df[active_df['is_anomaly'] == 1][export_cols]

st.sidebar.download_button(
    label="📥 Export Carbon & Audit Report (CSV)",
    data=export_df.to_csv(index=False).encode('utf-8'),
    file_name=f"YUKTHI_DigitalTwin_Report_{selected_chiller}.csv",
    mime="text/csv"
)

# =====================================================================
# 7. SPLIT WORKSPACE INTERFACE
# =====================================================================

col_left, col_right = st.columns([13, 11])

# --- LEFT PANEL: TIME-SERIES VISUAL ---
with col_left:
    st.subheader(f"📈 Real-Time Digital Twin Telemetry — {selected_chiller}")
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=active_df['timestamp'],
        y=active_df['Chiller Energy Consumption (kWh)'],
        mode='lines',
        name='Actual Consumption (kWh)',
        line=dict(color='#00f2fe', width=1.8)
    ))

    if 'digital_twin_baseline_kwh' in active_df.columns:
        fig.add_trace(go.Scatter(
            x=active_df['timestamp'],
            y=active_df['digital_twin_baseline_kwh'],
            mode='lines',
            name='Digital Twin Baseline',
            line=dict(color='#10b981', width=1.5, dash='dash')
        ))

    anom_points = active_df[active_df['is_anomaly'] == 1]
    fig.add_trace(go.Scatter(
        x=anom_points['timestamp'],
        y=anom_points['Chiller Energy Consumption (kWh)'],
        mode='markers',
        name=f'Contextual Anomaly (s > {sensitivity_threshold:.2f})',
        marker=dict(color='#ef4444', size=8, symbol='circle-open', line=dict(width=2))
    ))

    fig.update_layout(
        height=450,
        width=780,
        template='plotly_dark',
        paper_bgcolor='#080c14',
        plot_bgcolor='#0f172a',
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.12, x=0.0)
    )

    st.plotly_chart(fig, use_container_width=False)

# --- RIGHT PANEL: PRESCRIPTIVE OPERATIONS MATRIX ---
with col_right:
    st.subheader("🎯 Prescriptive Operations Matrix")
    
    asset_profiles = {
        'CHILLER-01': ('🔥 Primary Thermal Workhorse', 'High sensitivity to peak thermal load surges and ambient weather strain.'),
        'CHILLER-02': ('⚡ High-Capacity Inverter Unit', 'Susceptible to power factor fluctuations and efficiency drops.'),
        'CHILLER-03': ('💧 Hydronic Loop Regulator', 'Monitors chilled water differential pressures and flow rate stability.')
    }
    profile_title, profile_desc = asset_profiles.get(selected_chiller, ('Standard Chiller', 'Monitored asset'))
    st.info(f"**Asset Profile:** {profile_title}\n\n*{profile_desc}*")

    st.markdown("#### 🧠 Explainable Root Cause Attribution (XAI)")
    
    latest_anomaly_rows = active_df[active_df['is_anomaly'] == 1]
    sample_driver_str = latest_anomaly_rows['driver_matrix'].iloc[-1] if not latest_anomaly_rows.empty else active_df['driver_matrix'].iloc[-1]
        
    drivers = str(sample_driver_str).split('|')
    for driver in drivers:
        if ':' in driver:
            feat_name, pct_str = driver.split(':')
            pct_val = int(pct_str)
            st.write(f"**{feat_name}** — `{pct_val}%` Attribution")
            st.progress(min(1.0, max(0.0, pct_val / 100.0)))

    st.divider()
    st.markdown("#### 🛠️ Evidence-Based Engineering Procedures")
    
    displayed_codes = set()
    for driver in drivers:
        if ':' in driver:
            feat_name, pct_str = driver.split(':')
            pct_val = int(pct_str)
            
            current_val = active_df[feat_name].iloc[-1] if not active_df.empty else 0
            normal_baseline = active_df[feat_name].mean() if not active_df.empty else 0
            delta = current_val - normal_baseline
            
            if pct_val >= 50:
                risk_class = "risk-high"
                urgency = "<span style='color:#ef4444; font-weight:bold;'>HIGH RISK</span>"
            elif pct_val >= 25:
                risk_class = "risk-med"
                urgency = "<span style='color:#f59e0b; font-weight:bold;'>MEDIUM RISK</span>"
            else:
                risk_class = "risk-low"
                urgency = "<span style='color:#10b981; font-weight:bold;'>LOW RISK</span>"

            if ('Energy' in feat_name or 'kWh' in feat_name) and 'EN-01' not in displayed_codes:
                displayed_codes.add('EN-01')
                st.markdown(f"""
                <div class="directive-card {risk_class}">
                    <span class="code-badge">Code EN-01</span> [{urgency}] <b>Compressor Inverter Inefficiency</b><br/>
                    <b>XAI Delta:</b> Reduce consumption by <code>{abs(delta):.1f} kWh</code> toward baseline (<code>{normal_baseline:.1f} kWh</code>).<br/>
                    <b>Remediation:</b> Calibrate motor inverter thermal limits and audit compressor staging logic.
                </div>
                """, unsafe_allow_html=True)
                
            elif ('Water' in feat_name or 'Rate' in feat_name) and 'FL-04' not in displayed_codes:
                displayed_codes.add('FL-04')
                st.markdown(f"""
                <div class="directive-card {risk_class}">
                    <span class="code-badge">Code FL-04</span> [{urgency}] <b>Hydronic Loop Flow Restriction</b><br/>
                    <b>XAI Delta:</b> Adjust flow rate by <code>{abs(delta):.1f} L/sec</code> toward baseline (<code>{normal_baseline:.1f} L/sec</code>).<br/>
                    <b>Remediation:</b> Inspect differential pressure sensors and balance chilled-water loop control valves.
                </div>
                """, unsafe_allow_html=True)

            elif ('Temperature' in feat_name or 'Temp' in feat_name or 'Dew' in feat_name) and 'TM-02' not in displayed_codes:
                displayed_codes.add('TM-02')
                st.markdown(f"""
                <div class="directive-card {risk_class}">
                    <span class="code-badge">Code TM-02</span> [{urgency}] <b>Condenser Heat Exchanger Fouling</b><br/>
                    <b>XAI Delta:</b> Normalize temperature by <code>{abs(delta):.1f} °C</code> toward baseline (<code>{normal_baseline:.1f} °C</code>).<br/>
                    <b>Remediation:</b> Descale condenser tubes and inspect cooling tower airflow distribution fans.
                </div>
                """, unsafe_allow_html=True)
                
            elif ('Load' in feat_name or 'RT' in feat_name) and 'LD-03' not in displayed_codes:
                displayed_codes.add('LD-03')
                st.markdown(f"""
                <div class="directive-card {risk_class}">
                    <span class="code-badge">Code LD-03</span> [{urgency}] <b>Building Thermal Load Spike</b><br/>
                    <b>XAI Delta:</b> Re-balance load by <code>{abs(delta):.1f} RT</code> toward baseline (<code>{normal_baseline:.1f} RT</code>).<br/>
                    <b>Remediation:</b> Re-index air handler unit (AHU) mixing dampers and chilled water supply setpoints.
                </div>
                """, unsafe_allow_html=True)

# Incident Table Below Workspace
st.divider()
st.markdown("### 🚨 High-Severity Digital Twin Incident Log")
top_incidents = active_df[active_df['is_anomaly'] == 1].sort_values(by='anomaly_score', ascending=False).head(5)

log_cols = [c for c in ['timestamp', 'Chiller Energy Consumption (kWh)', 'digital_twin_baseline_kwh', 'anomaly_score', 'carbon_leakage_kg', 'financial_leakage_inr', 'driver_matrix'] if c in active_df.columns]

st.dataframe(
    top_incidents[log_cols],
    hide_index=True,
    use_container_width=True
)