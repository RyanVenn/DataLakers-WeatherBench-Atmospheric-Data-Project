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
emdat["Total Damage ('000 US$)"] = pd.to_numeric(
    emdat["Total Damage ('000 US$)"], errors="coerce"
)

emdat = emdat.dropna(subset=["Latitude", "Longitude"])

emdat_gdf = gpd.GeoDataFrame(
    emdat,
    geometry=gpd.points_from_xy(emdat["Longitude"], emdat["Latitude"]),
    crs="EPSG:4326"
)


gdf_proj = gdf.to_crs(epsg=3857)
emdat_proj = emdat_gdf.to_crs(epsg=3857)

gdf_proj["geometry"] = gdf_proj.geometry.buffer(100000)

joined = gpd.sjoin(emdat_proj, gdf_proj, predicate="within")

damage = joined.groupby("ID_UC_G0")["Total Damage ('000 US$)"].sum()

gdf["total_damage"] = gdf["ID_UC_G0"].map(damage).fillna(0)

plt.figure(figsize=(10,6))

plt.scatter(
    gdf["gdp_per_capita"],
    gdf["total_damage"],
    alpha=0.5
)

plt.xlabel("GDP per Capita")
plt.ylabel("Total Disaster Damage")
plt.title("Wealth vs Disaster Damage Exposure")

plt.grid(alpha=0.2)

plt.savefig("data_visualisation/new_york/visualisations/gdp_vs_damage.png", dpi=300)
plt.close()