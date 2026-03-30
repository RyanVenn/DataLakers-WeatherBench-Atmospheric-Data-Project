import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_ucdb

df = load_ucdb()

# Ensure numeric columns.
df["hazard_total"] = pd.to_numeric(df["hazard_total"], errors="coerce")
df["hdi"] = pd.to_numeric(df["hdi"], errors="coerce")

# Remove rows with invalid values.
df = df.dropna(subset=["hazard_total", "hdi"])

plt.figure(figsize=(10,7))

plt.hexbin(
    df["hazard_total"],
    df["hdi"],
    gridsize=20,
    # mincnt=1
)

plt.colorbar(label="Number of Cities")

plt.xlabel("Total Climate Hazard Events")
plt.ylabel("Human Development Index (HDI)")
plt.title("Human Development vs Climate Hazard Exposure")

plt.savefig("data_visualisation/ghs_ucdb/visualisations/hdi_vs_hazard.png", dpi=300)