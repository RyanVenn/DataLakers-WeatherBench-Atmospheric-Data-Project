import matplotlib.pyplot as plt
import pandas as pd

from weather_data_loader import load_weather_data

df = load_weather_data("datasets/weatherbench/*.csv")

ny = df[df["location_name"] == "New York (USA)"].copy()

# Monthly aggregation (reduces noise).
monthly = ny.resample("M", on="timestamp").agg({
    "temperature": "mean"
}).reset_index()

# Rolling smoothing (3-year window).
monthly["temp_smooth"] = monthly["temperature"].rolling(window=36).mean()

plt.figure(figsize=(10,6))
plt.plot(monthly["timestamp"], monthly["temperature"], alpha=0.2)
plt.plot(monthly["timestamp"], monthly["temp_smooth"])

plt.title("New York Temperature Trend (Smoothed)")
plt.xlabel("Year")
plt.ylabel("Temperature")

plt.show()