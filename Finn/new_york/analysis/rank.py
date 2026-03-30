import pandas as pd
from data_loader import load_ucdb

gdf = load_ucdb()

gdf["hazard_rank"] = gdf["hazard_total"].rank(ascending=False)
gdf["density_rank"] = gdf["density"].rank(ascending=False)

ny = gdf[gdf["city"].str.contains("New York", case=False, na=False)]

print(ny[[
    "city",
    "hazard_total",
    "hazard_rank",
    "density",
    "density_rank",
    "gdp_per_capita"
]])