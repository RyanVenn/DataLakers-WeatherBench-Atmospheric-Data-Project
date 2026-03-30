import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from data_loader import load_ucdb

gdf = load_ucdb()

gdf["population"] = pd.to_numeric(gdf["population"], errors="coerce")
gdf["hazard_total"] = pd.to_numeric(gdf["hazard_total"], errors="coerce")

df = gdf.dropna(subset=["population", "hazard_total"])

# Log population for readability.
df["log_population"] = np.log10(df["population"])

plt.figure(figsize=(10,7))

sns.boxplot(
    x="hazard_total",
    y="log_population",
    data=df
)

plt.xlabel("Number of Hazard Events")
plt.ylabel("Log Population")
plt.title("City Population vs Hazard Exposure")

plt.savefig(
    "data_visualisation/ghs_ucdb/visualisations/population_hazard_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)