import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from data_loader import load_ucdb

gdf = load_ucdb()

gdf["population"] = pd.to_numeric(gdf["population"], errors="coerce")
gdf["area_km2"] = pd.to_numeric(gdf["area_km2"], errors="coerce")

df = gdf.dropna(subset=["population", "area_km2"])

plt.figure(figsize=(10,7))

sns.scatterplot(
    x=df["population"],
    y=df["area_km2"],
    alpha=0.6,
    s=20
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Population (log scale)")
plt.ylabel("Urban Area km² (log scale)")
plt.title("Urban Scaling: Population vs Urban Area")

plt.savefig(
    "data_visualisation/ghs_ucdb/visualisations/population_area_scaling.png",
    dpi=300,
    bbox_inches="tight"
)