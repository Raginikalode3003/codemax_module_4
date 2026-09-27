# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 22:24:10 2026

@author: Raginikalode
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

# ==========================================
# STEP 1: FETCH FROM REST API (HTTP Request & JSON)
# ==========================================
print("[Step 1] Fetching data from API...")

# Free public API: Hourly 7-day temperature forecast for New York City
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "hourly": "temperature_2m",
    "timezone": "auto",
}

response = requests.get(url, params=params)

# Verify the request succeeded (HTTP 200 OK)
if response.status_code != 200:
    raise RuntimeError(f"API request failed with status: {response.status_code}")

payload = response.json()


# ==========================================
# STEP 2: LOAD & CLEAN DATA (Pandas DataFrame)
# ==========================================
print("[Step 2] Cleaning and structuring data into DataFrame...")

# Extract the hourly time series from the JSON payload
raw_data = {
    "timestamp": payload["hourly"]["time"],
    "temperature_c": payload["hourly"]["temperature_2m"],
}

df = pd.DataFrame(raw_data)

# Convert string timestamps to actual datetime objects
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Handling missing values (e.g., if any sensor reported null/NaN)
df = df.dropna(subset=["temperature_c"])

# Feature engineering: extract date and day of week
df["date"] = df["timestamp"].dt.date
df["day_name"] = df["timestamp"].dt.day_name()


# ==========================================
# STEP 3: ANALYZE & CALCULATE (NumPy & Pandas)
# ==========================================
print("[Step 3] Running statistical calculations...")

# Use NumPy for vector math: convert Celsius to Fahrenheit
# Formula: (C * 9/5) + 32
df["temperature_f"] = np.round((df["temperature_c"].to_numpy() * (9 / 5)) + 32, 2)

# Group by day to compute daily summary statistics
daily_summary = (
    df.groupby("day_name", sort=False)["temperature_c"]
    .agg(
        avg_temp="mean",
        min_temp="min",
        max_temp="max",
        temp_std=lambda x: np.std(x),  # NumPy standard deviation
    )
    .reset_index()
)

print("\n--- Daily Summary Report ---")
print(daily_summary.to_string(index=False))


# ==========================================
# STEP 4: VISUALIZE & REPORT (Seaborn & Matplotlib)
# ==========================================
print("\n[Step 4] Generating visual report...")

# Set Seaborn styling
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

# Subplot 1: Hourly trend line
sns.lineplot(
    data=df,
    x="timestamp",
    y="temperature_c",
    ax=axes[0],
    color="#2b5c8f",
    linewidth=2,
)
axes[0].set_title("7-Day Hourly Temperature Trend", fontsize=13, weight="bold")
axes[0].set_ylabel("Temp (°C)")
axes[0].set_xlabel("")

# Subplot 2: Daily average bar chart
sns.barplot(
    data=daily_summary,
    x="day_name",
    y="avg_temp",
    hue="day_name",
    palette="coolwarm",
    legend=False,
    ax=axes[1],
)
axes[1].set_title("Daily Average Temperature (°C)", fontsize=13, weight="bold")
axes[1].set_ylabel("Average Temp (°C)")
axes[1].set_xlabel("Day of Week")

# Adjust layout and display
plt.tight_layout()
plt.show()

# Optional: Export summary table to CSV
daily_summary.to_csv("weekly_weather_report.csv", index=False)
print("Saved clean summary to 'weekly_weather_report.csv'.")