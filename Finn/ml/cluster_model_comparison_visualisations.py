import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

INPUT_CSV = "DataLakers/Finn/ml/outputs/clustering_data/model_comparison.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/cluster_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Silhouette scores (left axis).
ax1.plot(df["k"], df["kmeans_silhouette"], 'o-', label="K-Means silhouette", color='blue')
ax1.plot(df["k"], df["gmm_silhouette"], 's-', label="GMM silhouette", color='orange')
ax1.set_xlabel("Number of clusters (K)")
ax1.set_ylabel("Silhouette score", color='black')
ax1.tick_params(axis='y')

# BIC (right axis).
ax2 = ax1.twinx()
ax2.plot(df["k"], df["gmm_bic"], 'd-', label="GMM BIC", color='green', alpha=0.7)
ax2.set_ylabel("Bayesian Information Criterion (BIC)", color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Highlight best K by BIC.
best_k = df.loc[df["gmm_bic"].idxmin(), "k"]
ax2.axvline(x=best_k, linestyle='--', color='red', alpha=0.5, label=f"Best BIC (K={int(best_k)})")

# Legends.
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

ax1.set_title("Model comparison: K-Means vs Gaussian Mixture Model")
ax1.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_vs_gmm_evaluation.png"), dpi=300)
plt.close()

print(f"Saved evaluation plot to {OUTPUT_DIR}/kmeans_vs_gmm_evaluation.png")