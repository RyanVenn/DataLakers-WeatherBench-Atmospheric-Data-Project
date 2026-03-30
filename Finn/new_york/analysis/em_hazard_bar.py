import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from data_loader import load_ucdb

gdf = load_ucdb()
ny = gdf[gdf["city"].str.contains("New York", case=False, na=False)]
if len(ny) == 0:
    raise ValueError("New York not found in UCDB")

ny = ny.iloc[0]

labels = [
    "Population",
    "Density",
    "GDP per capita",
    "Flood events",
    "Cyclone events"
]

values = [
    ny["population"],
    ny["density"],
    ny["gdp_per_capita"],
    ny["flood_events"],
    ny["cyclone_events"]
]

plt.figure(figsize=(10,6))
plt.bar(labels, values)

plt.title("New York: Urban + Hazard Profile")
plt.xticks(rotation=30)
plt.ylabel("Value")

Path("data_visualisation/new_york/visualisations").mkdir(exist_ok=True)
plt.savefig("data_visualisation/new_york/visualisations/ny_profile.png", dpi=300)
plt.close()