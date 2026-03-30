import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from data_loader import load_ucdb


gdf = load_ucdb()
gdf = gdf.dropna(subset=["density", "hazard_total"])
gdf["density_bin"] = pd.qcut(gdf["density"], 10, duplicates="drop")
grouped = gdf.groupby("density_bin")["hazard_total"].mean()


plt.figure(figsize=(10,6))
grouped.plot(kind="bar")

plt.title("Urban Density vs Hazard Exposure (Global Context)")
plt.ylabel("Average Hazard Events")
plt.xlabel("Density Decile")

Path("data_visualisation/new_york/visualisations").mkdir(exist_ok=True)
plt.savefig("data_visualisation/new_york/visualisations/density_vs_hazard.png", dpi=300)
plt.close()