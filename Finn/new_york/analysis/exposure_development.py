import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_ucdb

gdf = load_ucdb()

plt.figure(figsize=(10,7))

sns.scatterplot(
    data=gdf,
    x="gdp_per_capita",
    y="pop_exposed_100yr",
    alpha=0.3
)

ny = gdf[gdf["city"].str.contains("New York", case=False, na=False)]
plt.scatter(ny["gdp_per_capita"], ny["pop_exposed_100yr"], s=150)

plt.xscale("log")

plt.xlabel("GDP per Capita (log)")
plt.ylabel("Population Exposed (100yr events)")
plt.title("Wealth vs Hazard Exposure (New York Highlighted)")

plt.show()