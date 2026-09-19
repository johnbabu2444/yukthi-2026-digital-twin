# 🛡️ YUKTHI 2026 — Digital Twin Predictive Maintenance & Carbon-Efficiency Optimization Command Center

An enterprise-grade Industrial Digital Twin Command Center engineered for HVAC chiller plant monitoring, contextual anomaly detection, carbon leakage quantification (kg CO₂e), financial risk tracking (INR/hr), and Explainable AI (XAI) prescriptive maintenance.

---

## 📌 Executive Summary

Legacy industrial automation systems rely on static threshold alarms, which frequently trigger false alarms during peak ambient heat or miss subtle efficiency drops during off-peak hours.

This platform solves that limitation by introducing an **Asset-Aware Digital Twin Engine**. By comparing real-time operational telemetry against a dynamic synthetic thermodynamic baseline, the system isolates true contextual anomalies, quantifies excess carbon emissions, and delivers evidence-based engineering remediation codes (`EN-01`, `FL-04`, `TM-02`, `LD-03`, `ENV-05`).

---

## 🏗️ Technical Architecture & ML Core

### 1. Zero-Dependency Pure-NumPy Isolation Forest
The backend anomaly detection engine is implemented in pure Python/NumPy without relying on scikit-learn or external compiled C-extensions. It computes exact theoretical binary partition path lengths $c(n)$ and anomaly scores $s(x,n)$:

$$c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n}$$

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

### 2. Dynamic Synthetic Optimization Baseline (Digital Twin)
`detect_anomalies.py` programmatically calculates the ideal thermodynamic energy signature given current building thermal loads (RT) and ambient weather conditions:

$$\text{Ideal kWh} = \text{Load (RT)} \times 0.55 \times \left(1.0 + \max(0, \text{Temp}_{\text{outside}} - 70) \times 0.008\right)$$

### 3. Carbon & Sustainability Metrics
- **Excess Carbon Footprint (kg CO₂e):** Calculated using Indian commercial power grid intensity factors (~0.716 kg CO₂e / kWh).
- **Active Financial Leakage (INR/hr):** Quantified using regional commercial electricity tariffs (~₹8.50 / kWh).

### 4. Explainable AI (XAI) Prescriptive Matrix
Multidimensional Z-deviation feature attribution is decomposed across running medians to map specific sensor anomalies directly to urgency-classified engineering directives (**HIGH**, **MEDIUM**, **LOW RISK**) with counterfactual target deltas.

---

## 📂 Repository Structure

```text
├── app.py                    # Streamlit Industrial Mission Control Dashboard
├── detect_anomalies.py       # Pure-NumPy Isolation Forest & Digital Twin Engine
├── preprocess.py             # Telemetry cleaning & feature engineering module
├── development_dataset.csv   # Raw IoT sensor telemetry feed
├── anomalies_detected.csv    # Processed ML artifact with XAI driver matrices
├── requirements.txt          # Production environment dependencies
└── README.md                 # System documentation & deployment guide
```

---

## 🚀 Local Setup & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/yukthi-2026-digital-twin.git
cd yukthi-2026-digital-twin
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Digital Twin Pipeline
Execute the detection engine to generate `anomalies_detected.csv`:
```bash
python detect_anomalies.py
```

### 4. Launch the Command Center Dashboard
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.
