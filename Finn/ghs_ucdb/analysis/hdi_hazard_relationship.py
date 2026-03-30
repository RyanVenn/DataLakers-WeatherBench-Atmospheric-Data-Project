import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import load_ucdb

gdf = load_ucdb()

gdf["hdi"] = pd.to_numeric(gdf["hdi"], errors="coerce")
gdf["hazard_total"] = pd.to_numeric(gdf["hazard_total"], errors="coerce")

df = gdf.dropna(subset=["hdi", "hazard_total"])

plt.figure(figsize=(10,7))

sns.boxplot(
    x="hazard_total",
    y="hdi",
    data=df
)

plt.xlabel("Number of Hazard Events")
plt.ylabel("Human Development Index")
plt.title("Human Development vs Climate Hazard Exposure")

plt.savefig(
    "data_visualisation/ghs_ucdb/visualisations/hdi_hazard_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)