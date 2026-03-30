import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from data_loader import load_ucdb

gdf = load_ucdb()
ny = gdf[gdf["city"].str.contains("New York", case=False, na=False)]

if len(ny) == 0:
    raise ValueError("New York not found")

ny = ny.iloc[0]

hazards = {
    "Flood": ny["flood_events"],
    "Drought": ny["drought_events"],
    "Cyclone": ny["cyclone_events"],
    "Wildfire": ny["wildfire_events"],
    "Landslide": ny["landslide_events"]
}

plt.figure(figsize=(8,6))
plt.bar(hazards.keys(), hazards.values())

plt.title("New York Hazard Composition (UCDB)")
plt.ylabel("Event Frequency")

Path("data_visualisation/new_york/visualisations").mkdir(exist_ok=True)
plt.savefig("data_visualisation/new_york/visualisations/ny_hazard_composition.png", dpi=300)
plt.close()