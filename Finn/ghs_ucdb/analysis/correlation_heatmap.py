import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_ucdb

gdf = load_ucdb()

cols = [
    "population",
    "density",
    "area_km2",
    "hazard_total",
    "temp_mean",
    "precipitation",
    "elevation",
    "gdp_per_capita",
    "hdi"
]

for c in cols:
    gdf[c] = pd.to_numeric(gdf[c], errors="coerce")

df = gdf[cols].dropna()

corr = df.corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("Urban Dataset Correlation Matrix")

plt.savefig(
    "data_visualisation/ghs_ucdb/visualisations/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)