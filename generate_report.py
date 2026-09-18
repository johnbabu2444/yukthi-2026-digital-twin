import os
import pandas as pd
from datetime import datetime

def build_executive_summary():
    """Generates a structured text summary for executive export."""
    df = pd.read_csv('anomalies_detected.csv')
    critical_df = df[df['is_anomaly'] == 1]
    
    total_anomalies = len(critical_df)
    total_carbon = df['carbon_leakage_kg'].sum()
    total_waste_inr = df['financial_leakage_inr'].sum()
    
    report_text = f"""
================================================================================
YUKTHI 2026: DIGITAL TWIN EXECUTIVE INCIDENT & SUSTAINABILITY AUDIT REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. EXECUTIVE FLEET SUMMARY
--------------------------------------------------------------------------------
- Total Monitored Telemetry Records: {len(df):,}
- Total Contextual Anomalies Detected: {total_anomalies:,}
- Cumulative Excess Carbon Footprint: {total_carbon:,.2f} kg CO2e
- Total Operational Financial Waste: ₹{total_waste_inr:,.2f} INR

2. ASSET BREAKDOWN & INCIDENT COUNTS
--------------------------------------------------------------------------------
"""
    for eq in df['equipment_id'].unique():
        eq_anom = len(df[(df['equipment_id'] == eq) & (df['is_anomaly'] == 1)])
        report_text += f"- {eq}: {eq_anom} Critical Anomaly Events\n"
        
    report_text += f"""
3. HIGH-SEVERITY INCIDENTS LOG (TOP 5)
--------------------------------------------------------------------------------
"""
    top_5 = critical_df.sort_values(by='anomaly_score', ascending=False).head(5)
    for idx, row in top_5.iterrows():
        report_text += f"[{row['timestamp']}] {row['equipment_id']} | Score: {row['anomaly_score']:.3f} | Actual: {row['Chiller Energy Consumption (kWh)']} kWh | Drivers: {row['driver_matrix']}\n"
        
    report_text += """
================================================================================
END OF REPORT - YUKTHI 2026 DIGITAL TWIN COMMAND CENTER
================================================================================
"""
    with open('Executive_Audit_Report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    print("Executive Audit Report successfully generated at Executive_Audit_Report.txt")
    return report_text

if __name__ == "__main__":
    build_executive_summary()