import time
import pandas as pd
import numpy as np
from datetime import datetime

def append_live_telemetry():
    """Simulates real-time IoT telemetry ingestion by appending new time-series points."""
    df = pd.read_csv('development_dataset.csv')
    
    last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
    new_timestamp = (last_timestamp + pd.Timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
    
    for eq in ['CHILLER-01', 'CHILLER-02', 'CHILLER-03']:
        inject_anomaly = np.random.rand() > 0.7
        
        base_load = np.random.uniform(500, 900)
        base_energy = base_load * (0.95 if inject_anomaly else 0.52)
        
        new_row = {
            'timestamp': new_timestamp,
            'equipment_id': eq,
            'Chilled Water Rate (L/sec)': np.random.uniform(40, 100),
            'Cooling Water Temperature (C)': np.random.uniform(20, 35),
            'Building Load (RT)': base_load,
            'Chiller Energy Consumption (kWh)': base_energy,
            'Outside Temperature (F)': np.random.uniform(65, 95),
            'Dew Point (F)': np.random.uniform(50, 70),
            'Humidity (%)': np.random.uniform(40, 90),
            'Wind Speed (mph)': np.random.uniform(2, 15),
            'Pressure (in)': np.random.uniform(29.5, 30.2)
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    df.to_csv('development_dataset.csv', index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Ingested live IoT telemetry packet for all units!")

if __name__ == "__main__":
    print("Starting Live IoT Telemetry Simulator (Press Ctrl+C to stop)...")
    while True:
        append_live_telemetry()
        time.sleep(3)