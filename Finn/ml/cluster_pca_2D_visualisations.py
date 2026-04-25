import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})

INPUT_CLUSTERED = "DataLakers/Finn/ml/outputs/clustering_data/clustered_data_10.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/cluster_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "temp_mean_K",
    "wind_speed_mean_ms",
    "precip_total_mm",
    "population",
    "built_up_area_m2",
    "gdp_ppp",
    "hdi",
    "pop_exposed_flood_10yr",
    "co2_emissions_ton",
    "ghg_emissions_ton",
    "nox_emissions_ton",
    "pm25_emissions_ton"
]

# Which cluster column to use (should be "gmm_cluster" or "kmeans_cluster").
CLUSTER_COL = "gmm_cluster"


df = pd.read_csv(INPUT_CLUSTERED)
# Ensure year is integer.
df["year"] = df["year"].astype(int)
# Drop any rows with missing cluster labels (should not happen).
df = df.dropna(subset=[CLUSTER_COL])


X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA coordinates to dataframe.
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# Explained variance for axis labels.
var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100


def draw_ellipse(ax, x, y, n_std=2.0, facecolor='none', **kwargs):
    """
    Draw an ellipse based on the covariance of (x, y) data.
    n_std = number of standard deviations (2 = ~95% confidence).
    """
    cov = np.cov(x, y)
    mean = np.mean(x), np.mean(y)
    # Eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)


fig, ax = plt.subplots(figsize=(14, 10))

# Get unique clusters.
clusters = sorted(df[CLUSTER_COL].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

# Draw city trajectories (lines connecting years).
for city, group in df.groupby("city"):
    group = group.sort_values("year")
    ax.plot(group["PC1"], group["PC2"], 
            linestyle='-', linewidth=1, alpha=0.4, 
            color='gray', zorder=1)

# Scatter points coloured by cluster.
for cl, color in zip(clusters, colors):
    subset = df[df[CLUSTER_COL] == cl]
    ax.scatter(subset["PC1"], subset["PC2"],
               c=[color], label=f"Cluster {int(cl)}", 
               s=30, alpha=0.7, edgecolors='k', linewidth=0.3, zorder=2)

# Draw confidence ellipses (2 standard deviations).
for cl, color in zip(clusters, colors):
    subset = df[df[CLUSTER_COL] == cl]
    if len(subset) >= 4: # Need enough points for stable covariance.
        draw_ellipse(ax, subset["PC1"], subset["PC2"], n_std=2, edgecolor=color, facecolor='none', linewidth=2, linestyle='--', zorder=3)

# Annotate city names (place at the mean of each city's points).
city_centroids = df.groupby("city")[["PC1", "PC2"]].mean()
for city, (pc1, pc2) in city_centroids.iterrows():
    ax.annotate(city, xy=(pc1, pc2), xytext=(5, 5),
                textcoords='offset points', fontsize=16, 
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

ax.set_xlabel(f"Principal Component 1", fontsize=16)
ax.set_ylabel(f"Principal Component 2", fontsize=16)
ax.set_title("GMM Clusters in PCA Space", fontsize=18)
ax.legend(loc='lower right')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_clusters_with_trajectories_gmm.png"), dpi=300)
plt.close()

print(f"PCA plot saved to {OUTPUT_DIR}/pca_clusters_with_trajectories_gmm.png")