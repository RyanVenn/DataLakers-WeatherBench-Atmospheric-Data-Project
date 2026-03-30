import pandas as pd
import geopandas as gpd

DATA_PATH = "datasets/ghs_ucdb/ucdb_merged.csv"
GPKG_PATH = "datasets/ghs_ucdb/GHS_UCDB_GLOBE_R2024A.gpkg"


def load_ucdb():
    # Load attribute table.
    df = pd.read_csv(DATA_PATH)

    # Load spatial dataset.
    gdf = gpd.read_file(GPKG_PATH, layer="UC_centroids")

    # Merge using UCDB ID.
    gdf = gdf.merge(df, on="ID_UC_G0")

    return gdf