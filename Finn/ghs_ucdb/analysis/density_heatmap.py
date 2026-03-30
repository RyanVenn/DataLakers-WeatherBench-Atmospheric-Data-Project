import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
import numpy as np

from data_loader import load_ucdb

gdf = load_ucdb()

# Project to Web Mercator.
gdf = gdf.to_crs(epsg=3857)

centroids = gdf.geometry.centroid
x = centroids.x
y = centroids.y

weights = np.log1p(gdf["population"])

fig, ax = plt.subplots(figsize=(14,8))

# Set map extent.
ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2])
ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3])

# Add basemap.
cx.add_basemap(
    ax,
    source=cx.providers.CartoDB.Positron,
    zoom=2
)

# Hexbin density overlay.
hb = ax.hexbin(
    x,
    y,
    C=weights,
    reduce_C_function=np.sum,
    gridsize=200,
    cmap="magma",
    alpha=0.7,
    mincnt=1
)

# Colour bar.
cb = plt.colorbar(hb, ax=ax)
cb.set_label("Urban Population Density (log scale)")

ax.set_title("Global Urban Population Density")
ax.axis("off")

plt.savefig(
    "data_visualisation/ghs_ucdb/visualisations/density_hexbin.png",
    dpi=300,
    bbox_inches="tight"
)