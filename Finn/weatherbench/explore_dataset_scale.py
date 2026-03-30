import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_weather_data


path = sys.argv[1] if len(sys.argv) > 1 else "datasets/weatherbench/*.csv"  # Default path pattern if not provided as an argument.

df = load_weather_data(path)

print("\n==== DATASET OVERVIEW ====\n")

print("Total rows:", len(df))
print("Total locations: ", df["location"].nunique())
print("Time span: ", df["timestamp"].min(), " to ", df["timestamp"].max())

# Rows per file/location.
rows_per_location = df.groupby("location").size().sort_values(ascending=False)
print("\nRows per location: \n")
print(rows_per_location)
rows_per_location.to_csv("data_visualisation/scale/rows_per_location.csv")

# Records per year.
records_per_year = df.groupby("year").size()
print("\nRecords per year: \n")
print(records_per_year)
records_per_year.to_csv("data_visualisation/scale/records_per_year.csv")

# Sampling interval check.
sampling = df.sort_values("timestamp").groupby("location")["timestamp"].diff().dropna()

print("\nSampling interval stats: ")
print(sampling.describe())


# =========================
# VISUALISATIONS
# =========================

sns.set_style("whitegrid")

# Records per location.
plt.figure(figsize=(10,6))
rows_per_location.plot(kind="bar")
plt.title("Records per Location")
plt.ylabel("Row Count")
plt.tight_layout()
plt.savefig("data_visualisation/scale/records_per_location.png")
plt.close()

# Histogram of years.
plt.figure(figsize=(12,6))
records_per_year.plot(kind="bar")
plt.title("Records per Year")
plt.xlabel("Year")
plt.ylabel("Row Count")
plt.tight_layout()
plt.savefig("data_visualisation/scale/records_per_year_bar.png")
plt.close()


# Records per year line plot.
plt.figure(figsize=(12,6))
records_per_year.plot()
plt.title("Records per Year")
plt.xlabel("Year")
plt.ylabel("Row Count")
plt.tight_layout()
plt.savefig("data_visualisation/scale/records_per_year.png")
plt.close()