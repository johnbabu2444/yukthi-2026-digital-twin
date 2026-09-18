import pandas as pd
import numpy as np
from preprocess import clean_and_prepare_data

# =====================================================================
# 1. PURE-PYTHON ISOLATION FOREST ARCHITECTURE
# =====================================================================

def c_factor(n):
    """Normalized average path length factor c(n)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0

    def fit(self, X, current_depth=0):
        self.size = len(X)
        if current_depth >= self.max_depth or self.size <= 1:
            return

        n_features = X.shape[1]
        self.split_feature = np.random.randint(0, n_features)
        feat_vals = X[:, self.split_feature]
        
        min_val, max_val = feat_vals.min(), feat_vals.max()
        if min_val == max_val:
            return

        self.split_value = np.random.uniform(min_val, max_val)
        left_mask = feat_vals < self.split_value
        
        self.left = IsolationTree(self.max_depth)
        self.left.fit(X[left_mask], current_depth + 1)
        
        self.right = IsolationTree(self.max_depth)
        self.right.fit(X[~left_mask], current_depth + 1)

    def path_length(self, x, current_depth=0):
        if self.left is None or self.right is None or self.split_feature is None:
            return current_depth + c_factor(self.size)
        
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, current_depth + 1)
        else:
            return self.right.path_length(x, current_depth + 1)

class IsolationForestEngine:
    def __init__(self, n_estimators=40, max_samples=128):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []

    def fit(self, X):
        n_samples = len(X)
        max_depth = int(np.ceil(np.log2(max(self.max_samples, 2))))
        self.trees = []
        
        for _ in range(self.n_estimators):
            sample_size = min(self.max_samples, n_samples)
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            tree = IsolationTree(max_depth)
            tree.fit(X[indices])
            self.trees.append(tree)

    def compute_scores(self, X):
        paths = np.zeros((len(X), len(self.trees)))
        for t_idx, tree in enumerate(self.trees):
            for i in range(len(X)):
                paths[i, t_idx] = tree.path_length(X[i])
        
        avg_paths = paths.mean(axis=1)
        c = c_factor(self.max_samples)
        scores = 2.0 ** (- (avg_paths / c))
        return scores


# =====================================================================
# 2. DIGITAL TWIN SIMULATION & EXPLAINABLE ROOT CAUSE ATTRIBUTION
# =====================================================================

def compute_digital_twin_baseline(df):
    """
    Programmatically calculates the Dynamic Synthetic Optimization Baseline
    (ideal energy signature given building load, flow rate, and ambient weather).
    Ideal Chiller Efficiency Model: ~0.55 kWh per RT + ambient temperature modifier.
    """
    load_rt = df['Building Load (RT)']
    outside_temp = df['Outside Temperature (F)']
    
    # Thermodynamic efficiency coefficient curves
    temp_factor = 1.0 + ((outside_temp - 70.0).clip(lower=0) * 0.008)
    ideal_energy_kwh = load_rt * 0.55 * temp_factor
    return ideal_energy_kwh

def compute_feature_contributions(df_subset, feature_cols):
    """
    Calculates absolute Z-deviations against running medians to construct
    pipe-separated responsibility vector matrix strings.
    """
    X = df_subset[feature_cols].values
    medians = np.median(X, axis=0)
    mads = np.median(np.abs(X - medians), axis=0) + 1e-5
    
    z_deviations = np.abs(X - medians) / mads
    sum_devs = z_deviations.sum(axis=1, keepdims=True) + 1e-5
    contributions = (z_deviations / sum_devs) * 100
    
    contribution_strings = []
    for row in contributions:
        sorted_indices = np.argsort(row)[::-1]
        top_contribs = []
        for idx in sorted_indices[:3]:
            feat_name = feature_cols[idx]
            pct = int(round(row[idx]))
            top_contribs.append(f"{feat_name}:{pct}")
        contribution_strings.append("|".join(top_contribs))
        
    return contribution_strings


# =====================================================================
# 3. PIPELINE EXECUTION
# =====================================================================

def run_detection_pipeline():
    print("Ingesting and Cleaning Telemetry Feed...")
    df = clean_and_prepare_data('development_dataset.csv')
    
    feature_cols = [
        'Chilled Water Rate (L/sec)', 'Cooling Water Temperature (C)',
        'Building Load (RT)', 'Chiller Energy Consumption (kWh)',
        'Outside Temperature (F)', 'Dew Point (F)', 'Humidity (%)',
        'Wind Speed (mph)', 'Pressure (in)', 'Energy_per_Load', 'Energy_Rolling_3h'
    ]

    # Initialize Digital Twin Data Model
    df['digital_twin_baseline_kwh'] = compute_digital_twin_baseline(df)
    df['anomaly_score'] = 0.0
    df['is_anomaly'] = 0
    df['driver_matrix'] = ""

    # Asset-specific sensitivity mapping
    threshold_map = {
        'CHILLER-01': 0.53,
        'CHILLER-02': 0.58,
        'CHILLER-03': 0.55
    }

    print("Fitting Digital Twin Isolation Forest per Equipment Asset...")
    for equipment in df['equipment_id'].unique():
        eq_mask = df['equipment_id'] == equipment
        df_subset = df.loc[eq_mask].copy()
        eq_data = df_subset[feature_cols].values
        
        model = IsolationForestEngine(n_estimators=40, max_samples=128)
        model.fit(eq_data)
        
        scores = model.compute_scores(eq_data)
        df.loc[eq_mask, 'anomaly_score'] = scores
        
        unit_cutoff = threshold_map.get(equipment, 0.55)
        df.loc[eq_mask, 'is_anomaly'] = (scores > unit_cutoff).astype(int)
        
        drivers = compute_feature_contributions(df_subset, feature_cols)
        df.loc[eq_mask, 'driver_matrix'] = drivers

    # --- CARBON & SUSTAINABILITY LAYER ---
    # Excess Energy = Actual Consumption - Digital Twin Baseline
    df['excess_kwh'] = (df['Chiller Energy Consumption (kWh)'] - df['digital_twin_baseline_kwh']).clip(lower=0)
    
    # Commercial Grid Electricity Carbon Intensity in India (~0.716 kg CO2e / kWh)
    df['carbon_leakage_kg'] = df['excess_kwh'] * 0.716
    
    # Commercial Electricity Tariff (~INR 8.50 / kWh)
    df['financial_leakage_inr'] = df['excess_kwh'] * 8.50

    # --- PROACTIVE DEGRADATION FORECASTING ---
    # Compound Degradation Vector: Exponential rolling anomaly drift over 6-step time windows
    df['mechanical_integrity'] = 100.0 * (1.0 - df['anomaly_score'])
    df['degradation_risk_vector'] = df.groupby('equipment_id')['anomaly_score'].transform(
        lambda x: x.ewm(span=6, adjust=False).mean() * 100.0
    )

    df.to_csv('anomalies_detected.csv', index=False)
    print("\n=== DIGITAL TWIN PIPELINE PROCESSING COMPLETE ===")
    print("Summary Output Matrix saved to anomalies_detected.csv")
    return df

if __name__ == "__main__":
    run_detection_pipeline()