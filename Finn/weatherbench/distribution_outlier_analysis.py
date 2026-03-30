import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_weather_data


path = sys.argv[1] if len(sys.argv) > 1 else "datasets/weatherbench/*.csv"  # Default path pattern if not provided as an argument.
df = load_weather_data(path)

sns.set_style("whitegrid")

variables = [
    "temperature",
    "precipitation",
    "u_wind",
    "v_wind"
]


# Histograms + KDE (Kernel Density Estimation).
for var in variables:
    plt.figure(figsize=(10,6))

    if var == "precipitation":
        sns.histplot(df[var][df[var] > 0], bins=100)
        plt.yscale("log")
        plt.title("Precipitation Distribution (Non-zero values)")
    else:
        sns.histplot(df[var], kde=True, bins=100)
        plt.title(f"{var} distribution")

    plt.tight_layout()
    plt.savefig(f"data_visualisation/distribution_analysis/{var}_histogram.png")
    plt.close()


# Boxplots by location.
for var in variables:
    plt.figure(figsize=(18,10))

    if var == "precipitation":
        sns.boxplot(data=df, x="location", y="precipitation_mm")
        plt.xticks(rotation=45)
        plt.title(f"Precipitation Distribution by location")
    else:
        sns.boxplot(data=df, x="location", y=var)
        plt.xticks(rotation=45)
        plt.title(f"{var} by location")
    
    plt.tight_layout()
    plt.savefig(f"data_visualisation/distribution_analysis/{var}_box_location.png")
    plt.close()


# Precipitation occurrence heatmap.

# Rain event indicator
df["rain_event"] = df["precipitation"] > 0

# Probability of rain per month per location.
rain_heatmap = (
    df.groupby(["location", "month"])["rain_event"]
    .mean()
    .unstack()
)

plt.figure(figsize=(14,8))

sns.heatmap(
    rain_heatmap,
    cmap="Blues",
    linewidths=0.3,
    linecolor="gray"
)

plt.title("Rainfall Occurrence Probability by Month and Location")
plt.xlabel("Month")
plt.ylabel("Location")

plt.tight_layout()

plt.savefig("data_visualisation/distribution_analysis/rain_occurrence_heatmap.png")

plt.close()

# Z-Score outliers.
for var in variables:
    z = np.abs((df[var] - df[var].mean()) / df[var].std())
    outliers = df[z > 3]
    print(f"\n{var} outliers:", len(outliers))


# IQR outliers.
for var in variables:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)]
    print(f"{var} IQR outliers:", len(outliers))