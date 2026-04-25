import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})


INPUT_FILE = "DataLakers/Finn/ml/outputs/hmm_final/feature_comparison.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/hmm_final_feature_analysis/plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define feature groups.
CLIMATE_FEATURES = [
    "temp_mean_K",
    "wind_speed_mean_ms",
    "precip_total_mm"
]

SOCIO_FEATURES = [
    "population",
    "built_up_area_m2",
    "gdp_ppp",
    "hdi",
    "pop_exposed_flood_10yr"
]

EMISSIONS_FEATURES = [
    "co2_emissions_ton",
    "ghg_emissions_ton",
    "nox_emissions_ton",
    "pm25_emissions_ton"
]


df = pd.read_csv(INPUT_FILE)

# Feature-level aggregation.
feature_summary = df.groupby("feature").agg({
    "transitions": "mean",
    "avg_state_duration": "mean",
    "self_transition_prob": "mean",
    "n_states": "mean"
}).reset_index()

# Sort for nicer plots.
feature_summary = feature_summary.sort_values("transitions", ascending=False)

# Transitions per feature.
plt.figure(figsize=(10, 5))
plt.bar(feature_summary["feature"], feature_summary["transitions"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Number of Transitions")
plt.title("HMM State Transitions by Feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "transitions_per_feature.png", dpi=300)
plt.close()

# State stability/persistence (self-transition probability).
plt.figure(figsize=(10, 5))
plt.bar(feature_summary["feature"], feature_summary["self_transition_prob"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Self-Transition Probability")
plt.title("State Persistence by Feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "persistence_per_feature.png", dpi=300)
plt.close()

# State duration.
plt.figure(figsize=(10, 5))
plt.bar(feature_summary["feature"], feature_summary["avg_state_duration"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average State Duration (years)")
plt.title("Average State Duration by Feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "duration_per_feature.png", dpi=300)
plt.close()

# Grouped comparison (climate vs socioeconomic vs emissions features).
def classify_feature(f):
    if f in CLIMATE_FEATURES:
        return "Climate"
    elif f in SOCIO_FEATURES:
        return "Socioeconomic"
    elif f in EMISSIONS_FEATURES:
        return "Emissions"
    else:
        return "Other"

df["category"] = df["feature"].apply(classify_feature)
group_summary = df.groupby("category").agg({
    "transitions": "mean",
    "avg_state_duration": "mean",
    "self_transition_prob": "mean"
}).reset_index()

x = np.arange(len(group_summary))
width = 0.25

plt.figure(figsize=(8, 5))

plt.bar(x - width, group_summary["transitions"], width, label="Transitions")
plt.bar(x, group_summary["avg_state_duration"], width, label="State Duration")
plt.bar(x + width, group_summary["self_transition_prob"], width, label="Persistence")

plt.xticks(x, group_summary["category"])
plt.ylabel("Value")
plt.title("HMM Behaviour by Feature Category")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "category_comparison.png", dpi=300)
plt.close()

# Heatmap (feature x city transitions).
pivot = df.pivot(index="feature", columns="city", values="transitions")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")

plt.title("State Transitions per Feature and City")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "heatmap_transitions.png", dpi=300)
plt.close()

# Hyderabad specific analysis (top features by transitions).
hyd = df[df["city"] == "Hyderabad"]
hyd = hyd.sort_values("transitions", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(hyd["feature"], hyd["transitions"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Transitions")
plt.title("Hyderabad: State Transitions by Feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "hyderabad_transitions.png", dpi=300)
plt.close()


print("Plots saved to:", OUTPUT_DIR)