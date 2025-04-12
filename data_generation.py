import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

def fetch_api_data(latitude=28.6139, longitude=77.2090, start_date="20240101", end_date="20240412"):
    # NASA Power API endpoint
    api_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOT,ALLSKY_SFC_SW_DWN,RH2M",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()["properties"]["parameter"]
        print("API Response Keys:", data.keys())  # Debug: Print available keys
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        # Fallback: Use local JSON file
        with open("fallback_5g_data.json", "r") as f:
            data = eval(f.read())  # Replace with json.load for safety in production
        return [{"T2M": 25, "PRECTOT": 0.1, "ALLSKY_SFC_SW_DWN": 200, "RH2M": 60}]  # Mock data
    return data

def preprocess_data(api_data):
    dates = list(api_data["T2M"].keys())  # Use T2M dates as base (assumed always present)
    df = pd.DataFrame({
        "date": dates,
        "latency": [float(api_data["T2M"][d]) * 10 for d in dates],  # Temperature (°C) * 10 → Latency (ms)
        "interference": [float(api_data.get("PRECTOT", {}).get(d, 0)) * 100 if float(api_data.get("PRECTOT", {}).get(d, 0)) > 0 else 0.1 for d in dates],  # Fallback to 0 if PRECTOT missing
        "power": [float(api_data["ALLSKY_SFC_SW_DWN"][d]) * 0.5 for d in dates],  # Radiation (W/m²) * 0.5 → Power (mW)
        "snr": [100 - float(api_data["RH2M"][d]) for d in dates],  # 100 - Humidity (%) → SNR (dB)
        "bandwidth": np.random.uniform(30, 80, len(dates))  # Synthetic bandwidth (Mbps)
    })
    # Add realistic synthetic targets based on inputs
    df["spectrum"] = df["bandwidth"] * np.random.uniform(0.1, 0.3, len(dates))  # Proportional to bandwidth
    df["allocated_power"] = df["power"] * np.random.uniform(0.5, 1.5, len(dates))  # Proportional to power

    # Save to CSV
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    df.to_csv("datasets/5g_data.csv", index=False)
    return df

if __name__ == "__main__":
    api_data = fetch_api_data()
    df = preprocess_data(api_data)
    print(f"Data saved to datasets/5g_data.csv with {len(df)} rows")