import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_loader import load_ucdb


df = load_ucdb()

plt.figure(figsize=(8,6))

sns.scatterplot(
    x="density",
    y="gdp_per_capita",
    data=df,
    alpha=0.6
)

plt.xscale("log")
plt.xlabel("Population Density")
plt.ylabel("GDP per capita")
plt.yticks(np.arange(0, 12000, 1000), labels=[str(x) for x in np.arange(0, 12000, 1000)])
plt.title("Urban Density vs Economic Output")

plt.savefig("data_visualisation/ghs_ucdb/visualisations/gdp_density.png", dpi=300)