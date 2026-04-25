import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT = "DataLakers/Finn/ml/outputs/clustering_data/clustered_data.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/cluster_analysis_unused/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

CLUSTER_COL = "gmm_cluster"


features = [
    "temp_mean_K",
    "wind_speed_mean_ms",
    "precip_total_mm",
    "population",
    "built_up_area_m2",
    "gdp_ppp",
    "hdi",
    "co2_emissions_ton",
    "ghg_emissions_ton",
    "nox_emissions_ton",
    "pm25_emissions_ton"
]

cluster_profile = df.groupby(CLUSTER_COL)[features].mean()

# Plot feature profiles.
for feature in features:
    plt.figure()
    cluster_profile[feature].plot(kind="bar")
    plt.title(f"{feature} by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{feature}_by_cluster.png")
    plt.close()


# Cluster distribution over time.
year_cluster = df.groupby(["year", CLUSTER_COL]).size().unstack(fill_value=0)
year_cluster_norm = year_cluster.div(year_cluster.sum(axis=1), axis=0)
year_cluster_norm.plot(figsize=(10,6))
plt.title("Cluster Distribution Over Time")
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_distribution_over_time.png")
plt.close()


# Transition analysis.

df = df.sort_values(["city", "year"])
transitions = {}
for city, group in df.groupby("city"):
    group = group.sort_values("year")
    clusters = group[CLUSTER_COL].values
    
    for i in range(len(clusters)-1):
        pair = (clusters[i], clusters[i+1])
        transitions[pair] = transitions.get(pair, 0) + 1

# Convert to matrix.
transition_matrix = pd.DataFrame(0, 
    index=sorted(df[CLUSTER_COL].unique()),
    columns=sorted(df[CLUSTER_COL].unique())
)

for (i, j), count in transitions.items():
    transition_matrix.loc[i, j] = count

transition_matrix.to_csv(f"{OUTPUT_DIR}/transition_matrix.csv")

# Plot heatmap of transitions.
plt.figure(figsize=(8,6))
plt.imshow(transition_matrix, aspect='auto')
plt.colorbar(label="Transitions")
plt.xticks(range(len(transition_matrix.columns)), transition_matrix.columns)
plt.yticks(range(len(transition_matrix.index)), transition_matrix.index)
plt.title("Cluster Transition Matrix")
plt.xlabel("Next Cluster")
plt.ylabel("Current Cluster")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/transition_matrix.png")
plt.close()


print("All cluster analysis outputs saved.")