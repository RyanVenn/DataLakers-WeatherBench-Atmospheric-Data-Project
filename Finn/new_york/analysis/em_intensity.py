import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from data_loader import load_ucdb


gdf = load_ucdb().to_crs(epsg=4326)

FILE = "datasets/em_dat/united_states.xlsx"
emdat = pd.read_excel(FILE)

emdat = emdat[emdat["Disaster Group"] == "Natural"]

emdat["Latitude"] = pd.to_numeric(emdat["Latitude"], errors="coerce")
emdat["Longitude"] = pd.to_numeric(emdat["Longitude"], errors="coerce")
emdat = emdat.dropna(subset=["Latitude", "Longitude"])

emdat_gdf = gpd.GeoDataFrame(
    emdat,
    geometry=gpd.points_from_xy(emdat["Longitude"], emdat["Latitude"]),
    crs="EPSG:4326"
)

# Spatial join.
gdf_proj = gdf.to_crs(epsg=3857)
emdat_proj = emdat_gdf.to_crs(epsg=3857)

gdf_proj["geometry"] = gdf_proj.geometry.buffer(100000)

joined = gpd.sjoin(emdat_proj, gdf_proj, predicate="within")

counts = joined.groupby("ID_UC_G0").size()
gdf["disaster_count"] = gdf["ID_UC_G0"].map(counts).fillna(0)

# Plot map.
gdf_plot = gdf.to_crs(epsg=3857)

plt.figure(figsize=(12,8))

gdf_plot.plot(
    column="disaster_count",
    cmap="viridis",
    markersize=5,
    legend=True
)

plt.title("Disaster Exposure by City")
plt.axis("off")

plt.savefig("data_visualisation/new_york/visualisations/disaster_map.png", dpi=300)
plt.close()