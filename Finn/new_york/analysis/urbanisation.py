import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_ucdb

gdf = load_ucdb()

plt.figure(figsize=(10,7))

sns.scatterplot(
    data=gdf,
    x="density",
    y="flood_events",
    alpha=0.3
)

# Highlight New York.
ny = gdf[gdf["city"].str.contains("New York", case=False, na=False)]
plt.scatter(ny["density"], ny["flood_events"], s=150)

plt.xlabel("Population Density")
plt.ylabel("Flood Events")
plt.title("Density vs Flood Risk (New York Highlighted)")

plt.show()