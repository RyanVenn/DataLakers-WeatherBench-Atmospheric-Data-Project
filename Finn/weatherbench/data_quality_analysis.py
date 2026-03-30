import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from data_loader import load_weather_data


path = sys.argv[1] if len(sys.argv) > 1 else "datasets/weatherbench/*.csv"  # Default path pattern if not provided as an argument.
df = load_weather_data(path)

print("\n==== DATA QUALITY ====\n")

# Missing values.
missing = df.isna().sum()

print("\nMissing values per column: \n")
print(missing)
missing.to_csv("data_visualisation/quality_analysis/missing_values_per_column.csv")

# Duplicates.
duplicates = df.duplicated().sum()
print("\nDuplicate rows: ", duplicates)

# Timestamp continuity per location.
print("\nChecking timestamp continuity...\n")
gaps = []
for loc, group in df.groupby("location"):
    group = group.sort_values("timestamp")
    diff = group["timestamp"].diff()
    discontinuities = diff[diff > pd.Timedelta(hours=1)]
    gaps.append({
        "location": loc,
        "num_gaps": len(discontinuities)
    })

gap_df = pd.DataFrame(gaps)
print(gap_df)
gap_df.to_csv("data_visualisation/quality_analysis/timestamp_gaps.csv", index=False)


# =================
# VISUALISATIONS
# =================

sns.set_style("whitegrid")

# Missing values heatmap.
plt.figure(figsize=(12,6))
msno.matrix(df.sample(50000))
plt.title("Missing Data Matrix (Sample)")
plt.savefig("data_visualisation/quality_analysis/missing_matrix.png")
plt.close()

# Missing values per variable.
plt.figure(figsize=(8,5))
missing.plot(kind="bar")
plt.title("Missing Values per Variable")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data_visualisation/quality_analysis/missing_values_bar.png")
plt.close()

# Missing values per location.
missing_location = df.groupby("location").apply(lambda x: x.isna().sum())
missing_location.to_csv("data_visualisation/quality_analysis/missing_values_by_location.csv")

plt.figure(figsize=(12,6))
sns.heatmap(missing_location, cmap="viridis")
plt.title("Missing Values by Location")
plt.tight_layout()
plt.savefig("data_visualisation/quality_analysis/missing_by_location_heatmap.png")
plt.close()