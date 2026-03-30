import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from data_loader import load_ucdb

gdf = load_ucdb()

gdf["density"] = pd.to_numeric(gdf["density"], errors="coerce")
gdf["temp_mean"] = pd.to_numeric(gdf["temp_mean"], errors="coerce")

df = gdf.dropna(subset=["density", "temp_mean"])

plt.figure(figsize=(10,7))

sns.scatterplot(
    x=df["temp_mean"],
    y=np.log10(df["density"]),
    alpha=0.6,
    s=20
)

plt.xlabel("Mean Temperature")
plt.ylabel("Log Urban Density")
plt.title("Climate vs Urban Density")

plt.savefig(
    "data_visualisation/ghs_ucdb/visualisations/climate_density.png",
    dpi=300,
    bbox_inches="tight"
)