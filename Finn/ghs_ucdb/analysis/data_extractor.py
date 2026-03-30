import pandas as pd
from pathlib import Path

INPUT_FILE = "datasets/ghs_ucdb/GHS_UCDB_GLOBE_R2024A.xlsx"
OUTPUT_FILE = "datasets/ghs_ucdb/ucdb_merged.csv"


def extract_ucdb():
    sheets = pd.read_excel(INPUT_FILE, sheet_name=None)

    general = sheets["GENERAL_CHARACTERISTICS"]
    hazard = sheets["HAZARD_RISK"]
    climate = sheets["CLIMATE"]
    exposure = sheets["EXPOSURE"]
    geography = sheets["GEOGRAPHY"]
    socio = sheets["SOCIOECONOMIC"]

    # Start with general characteristics.
    df = general[
        [
            "ID_UC_G0",
            "GC_UCN_MAI_2025",
            "GC_CNT_GAD_2025",
            "GC_POP_TOT_2025",
            "GC_UCA_KM2_2025",
        ]
    ].copy()

    # Hazard subset.
    hazard_cols = [
        "ID_UC_G0",
        "HZ_CEV_FLO_2015",
        "HZ_CEV_DRO_2015",
        "HZ_CEV_TCY_2015",
        "HZ_CEV_WLF_2015",
        "HZ_CEV_LAN_2015",
    ]

    df = df.merge(hazard[hazard_cols], on="ID_UC_G0", how="left")

    # Climate subset.
    climate_cols = [
        "ID_UC_G0",
        "CL_B01_CUR_2010",
        "CL_B07_CUR_2010",
        "CL_B12_CUR_2010",
    ]

    df = df.merge(climate[climate_cols], on="ID_UC_G0", how="left")

    # Exposure subset.
    exposure_cols = [
        "ID_UC_G0",
        "EX_010_POP_2020",
        "EX_100_POP_2020",
    ]

    df = df.merge(exposure[exposure_cols], on="ID_UC_G0", how="left")

    # Geography subset.
    geography_cols = [
        "ID_UC_G0",
        "GE_ELV_AVG_2025",
    ]

    df = df.merge(geography[geography_cols], on="ID_UC_G0", how="left")

    socio_cols = [
        "ID_UC_G0",
        "SC_GDP_AVG_2015",
        "SC_SEC_HDI_2015",
        "SC_SEC_LET_2015",
        "SC_SEC_SYT_2015"
    ]

    df = df.merge(socio[socio_cols], on="ID_UC_G0", how="left")


    # Rename columns to readable names.
    df = df.rename(
        columns={
            "GC_UCN_MAI_2025": "city",
            "GC_CNT_GAD_2025": "country",
            "GC_POP_TOT_2025": "population",
            "GC_UCA_KM2_2025": "area_km2",
            "CL_B01_CUR_2010": "temp_mean",
            "CL_B07_CUR_2010": "temp_range",
            "CL_B12_CUR_2010": "precipitation",
            "HZ_CEV_FLO_2015": "flood_events",
            "HZ_CEV_DRO_2015": "drought_events",
            "HZ_CEV_TCY_2015": "cyclone_events",
            "HZ_CEV_WLF_2015": "wildfire_events",
            "HZ_CEV_LAN_2015": "landslide_events",
            "EX_010_POP_2020": "pop_exposed_10yr",
            "EX_100_POP_2020": "pop_exposed_100yr",
            "GE_ELV_AVG_2025": "elevation",
            "SC_GDP_AVG_2015": "gdp_ppp",
            "SC_SEC_HDI_2015": "hdi",
            "SC_SEC_LET_2015": "life_expectancy",
            "SC_SEC_SYT_2015": "years_schooling",
        }
    )

    # Ensure hazard columns are numeric.
    hazard_numeric_cols = [
        "flood_events",
        "drought_events",
        "cyclone_events",
        "wildfire_events",
        "landslide_events",
    ]

    for col in hazard_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived variables.
    df["density"] = df["population"] / df["area_km2"]

    df["hazard_total"] = df[
        [
            "flood_events",
            "drought_events",
            "cyclone_events",
            "wildfire_events",
            "landslide_events",
        ]
    ].fillna(0).sum(axis=1)

    df["gdp_per_capita"] = df["gdp_ppp"] / df["population"]

    Path("datasets/ghs_ucdb").mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", df.shape)


if __name__ == "__main__":
    extract_ucdb()