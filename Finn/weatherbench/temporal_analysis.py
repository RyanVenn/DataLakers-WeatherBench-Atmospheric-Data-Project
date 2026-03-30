import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_weather_data


path = sys.argv[1] if len(sys.argv) > 1 else "datasets/weatherbench/*.csv"
df = load_weather_data(path)

sns.set_style("whitegrid")

df["month"] = df["timestamp"].dt.month
df["hour"] = df["timestamp"].dt.hour
df["year"] = df["timestamp"].dt.year


for location, loc_df in df.groupby("location_name"):

    safe_name = location.replace(" ", "_").replace("(", "").replace(")", "")

    # Monhtly seasonal patterns.
    monthly_temp = loc_df.groupby("month")["temperature"].mean()
    monthly_precip = loc_df.groupby("month")["precipitation_mm"].mean()
    monthly_wind = loc_df.groupby("month")["wind_speed"].mean()

    plt.figure(figsize=(10,6))
    monthly_temp.plot(marker="o")
    plt.title(f"Average Monthly Temperature - {location}")
    plt.ylabel("Temperature (K)")
    plt.savefig(f"data_visualisation/temporal/{safe_name}_monthly_temperature.png")
    plt.close()


    plt.figure(figsize=(10,6))
    monthly_precip.plot(marker="o")
    plt.title(f"Average Monthly Precipitation - {location}")
    plt.ylabel("Precipitation (mm)")
    plt.savefig(f"data_visualisation/temporal/{safe_name}_monthly_precipitation.png")
    plt.close()


    plt.figure(figsize=(10,6))
    monthly_wind.plot(marker="o")
    plt.title(f"Average Monthly Wind Speed - {location}")
    plt.ylabel("Wind Speed (m/s)")
    plt.savefig(f"data_visualisation/temporal/{safe_name}_monthly_wind.png")
    plt.close()


    # Diurnal (daily) cycle.
    hour_temp = loc_df.groupby("hour")["temperature"].mean()
    hour_wind = loc_df.groupby("hour")["wind_speed"].mean()

    plt.figure(figsize=(10,6))
    hour_temp.plot(marker="o")
    plt.title(f"Temperature by Hour - {location}")
    plt.ylabel("Temperature (K)")
    plt.savefig(f"data_visualisation/temporal/{safe_name}_hourly_temperature.png")
    plt.close()


    plt.figure(figsize=(10,6))
    hour_wind.plot(marker="o")
    plt.title(f"Wind Speed by Hour - {location}")
    plt.ylabel("Wind Speed (m/s)")
    plt.savefig(f"data_visualisation/temporal/{safe_name}_hourly_wind.png")
    plt.close()


    # Long term trends.
    yearly_temp = loc_df.groupby("year")["temperature"].mean()

    rolling = yearly_temp.rolling(5).mean()

    plt.figure(figsize=(12,6))
    yearly_temp.plot(label="Yearly")
    rolling.plot(label="5-Year Rolling")
    plt.legend()
    plt.title(f"Temperature Trend - {location}")
    plt.ylabel("Temperature (K)")
    plt.savefig(f"data_visualisation/temporal/{safe_name}_temperature_trend.png")
    plt.close()