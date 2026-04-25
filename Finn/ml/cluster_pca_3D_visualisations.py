import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os


INPUT_CLUSTERED = "DataLakers/Finn/ml/outputs/clustering_data/clustered_data_10.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/cluster_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "temp_mean_K", "wind_speed_mean_ms", "precip_total_mm",
    "population", "built_up_area_m2", "gdp_ppp", "hdi",
    "pop_exposed_flood_10yr", "co2_emissions_ton", "ghg_emissions_ton",
    "nox_emissions_ton", "pm25_emissions_ton"
]
CLUSTER_COL = "gmm_cluster"


# Load data and compute PCA (3 components).
df = pd.read_csv(INPUT_CLUSTERED)
df["year"] = df["year"].astype(int)
df = df.dropna(subset=[CLUSTER_COL])
X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
df["PC3"] = X_pca[:, 2]

var = pca.explained_variance_ratio_ * 100

# Pairplot of PC1, PC2, PC3.
pca_df = df[["PC1", "PC2", "PC3", CLUSTER_COL, "city"]].copy()
pca_df[CLUSTER_COL] = pca_df[CLUSTER_COL].astype(int)

g = sns.pairplot(pca_df, hue=CLUSTER_COL, vars=["PC1", "PC2", "PC3"], diag_kind='hist', palette='tab10', plot_kws={'s': 20, 'alpha': 0.6})
g.fig.suptitle(f"Pairplot of first three principal components (total variance = {sum(var[:3]):.1f}%)", y=1.02)
plt.savefig(os.path.join(OUTPUT_DIR, "3d_pca_pairplot.png"), dpi=300)
plt.close()

print(f"Pairplot saved to {OUTPUT_DIR}/3d_pca_pairplot.png")


# 3D scatter plot.
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

clusters = sorted(df[CLUSTER_COL].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

for cl, color in zip(clusters, colors):
    subset = df[df[CLUSTER_COL] == cl]
    ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"], c=[color], label=f"Cluster {cl}", s=30, alpha=0.7, edgecolors='k')

ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax.set_zlabel(f"PC3 ({var[2]:.1f}%)")
ax.set_title(f"3D PCA projection of GMM clusters (K={len(clusters)})")
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3d_pca_scatter.png"), dpi=300)
plt.close()

print(f"3D scatter saved to {OUTPUT_DIR}/3d_pca_scatter.png")