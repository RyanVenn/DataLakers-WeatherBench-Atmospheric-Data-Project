import matplotlib.pyplot as plt
import pandas as pd

from weather_data_loader import load_weather_data

df = load_weather_data("datasets/weatherbench/*.csv")

ny = df[df["location_name"] == "New York (USA)"].copy()

ny["year"] = ny["timestamp"].dt.year

# Define heatwave relative to long-term threshold.
threshold = ny["temperature"].quantile(0.98)

ny["heatwave"] = ny["temperature"] > threshold

yearly = ny.groupby("year")["heatwave"].sum().reset_index()
yearly["smooth"] = yearly["heatwave"].rolling(5).mean()

plt.figure(figsize=(10,6))
plt.plot(yearly["year"], yearly["heatwave"], alpha=0.3)
plt.plot(yearly["year"], yearly["smooth"])

plt.title("Heatwave Frequency (Smoothed)")
plt.xlabel("Year")
plt.ylabel("Count")

plt.show()