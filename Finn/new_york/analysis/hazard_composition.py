import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_ucdb

gdf = load_ucdb()

ny = gdf[gdf["city"].str.contains("New York", case=False, na=False)]

if ny.empty:
    raise ValueError("New York not found in dataset")

ny = ny.iloc[0]

hazards = {
    "Flood": ny.get("flood_events", 0),
    "Cyclone": ny.get("cyclone_events", 0),
    "Drought": ny.get("drought_events", 0),
    "Wildfire": ny.get("wildfire_events", 0),
    "Landslide": ny.get("landslide_events", 0)
}

# Replace NaN with 0
hazards = {k: (0 if pd.isna(v) else v) for k, v in hazards.items()}

plt.figure(figsize=(8,6))
plt.bar(list(hazards.keys()), list(hazards.values()))

plt.title("Hazard Profile - New York")
plt.ylabel("Event Count")

plt.show()