import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_ucdb


df = load_ucdb()

plt.figure(figsize=(8,6))

sns.histplot(df["population"], bins=500, log_scale=(True, False))

plt.xlabel("Population")
plt.ylabel("City count")
plt.title("Distribution of City Population (UCDB)")

plt.savefig("data_visualisation/ghs_ucdb/visualisations/population_distribution.png", dpi=300)