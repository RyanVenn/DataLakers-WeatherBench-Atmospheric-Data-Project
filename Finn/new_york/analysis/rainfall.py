import matplotlib.pyplot as plt
import pandas as pd

from weather_data_loader import load_weather_data

df = load_weather_data("datasets/weatherbench/*.csv")

ny = df[df["location_name"] == "New York (USA)"].copy()

ny["year"] = ny["timestamp"].dt.year

threshold = ny["precipitation_mm"].quantile(0.99)

ny["extreme_rain"] = ny["precipitation_mm"] > threshold

yearly = ny.groupby("year")["extreme_rain"].sum().reset_index()
yearly["smooth"] = yearly["extreme_rain"].rolling(5).mean()

plt.figure(figsize=(10,6))
plt.plot(yearly["year"], yearly["extreme_rain"], alpha=0.3)
plt.plot(yearly["year"], yearly["smooth"])

plt.title("Extreme Rainfall Events (Smoothed)")
plt.xlabel("Year")
plt.ylabel("Count")

plt.show()