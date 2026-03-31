import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

UCDB_FILE = "datasets/ghs_ucdb/GHS_UCDB_GLOBE_R2024A.xlsx"
WEATHER_PATTERN = "datasets/weatherbench/*.csv"
OUTPUT_FILE = "datasets/merged_all_locations_yearly_1990_2018.csv"
YEARS = list(range(1990, 2019))  # 1990-2018.

# Mapping from location code to (city_name, country) for UCDB matching.
WEATHER_TO_CITY = {
    "-33.5N_151E": ("Maroota", "Australia"),
    "-33.9N_18.5E": ("Cape Town", "South Africa"),
    "17.36N_78.5E": ("Hyderabad", "India"),
    "18N_283.2E": ("Kingston", "Jamaica"),
    "40.75N_286.01E": ("New York", "United States of America"),
    "41.9N_12.46E": ("Rome", "Italy"),
    "43.28N_5.39E": ("Marseille", "France"),
    "43.64N_280.63E": ("Toronto", "Canada"),
    "51.03N_245.94E": ("Calgary", "Canada"),
    "51.5N_359.9E": ("London", "United Kingdom"),
    "58.76N_265.83E": ("Churchill", "Canada"),
    "59.92N_10.75E": ("Oslo", "Norway"),
}


def extract_ucdb_yearly_series(df_sheet, prefix, years, id_col="ID_UC_G0", target_id=None):
    """Extract time series for a given ID and return as Series with year index."""
    series = {}
    for year in years:
        col = f"{prefix}_{year}"
        if col in df_sheet.columns:
            val = df_sheet.loc[df_sheet[id_col] == target_id, col].values
            series[year] = val[0] if len(val) > 0 else np.nan
    return pd.Series(series, name=prefix)

def load_weather_for_location(filepath, location_code):
    """Load a single weather file and compute yearly aggregates."""
    df = pd.read_csv(filepath, header=None, names=["year","month","day","hour","temperature","precipitation","u_wind","v_wind"])
    # Dervive wind speed from u and v components (they both have very similar distributions so we only need one).
    df["wind_speed"] = (df["u_wind"]**2 + df["v_wind"]**2)**0.5
    
    yearly = df.groupby("year").agg(
        temp_mean=("temperature", "mean"),
        precip_total_m=("precipitation", "sum"),        # Total precipitation in metres.
        wind_mean=("wind_speed", "mean")
    ).reset_index()
    # Convert to mm.
    yearly["precip_total_mm"] = yearly["precip_total_m"] * 1000
    # Drop the metre column to avoid duplication.
    yearly = yearly.drop(columns=["precip_total_m"])
    yearly["location_code"] = location_code
    # Add the city name and country from mapping.
    if location_code in WEATHER_TO_CITY:
        yearly["city"], yearly["country"] = WEATHER_TO_CITY[location_code]
    else:
        yearly["city"] = location_code
        yearly["country"] = "Unknown"
    return yearly

def get_ucdb_id_for_city(sheets, city, country):
    """Find UCDB ID for given city and country."""
    general = sheets["GENERAL_CHARACTERISTICS"]
    matches = general[(general["GC_UCN_MAI_2025"] == city) & (general["GC_CNT_GAD_2025"] == country)]
    if matches.empty:
        return None
    
    return matches.iloc[0]["ID_UC_G0"]

def get_ucdb_series_for_location(sheets, location_code):
    """Get UCDB time series for a location (if city matched)."""
    if location_code not in WEATHER_TO_CITY:
        return None
    
    city, country = WEATHER_TO_CITY[location_code]
    uc_id = get_ucdb_id_for_city(sheets, city, country)
    if uc_id is None:
        print(f"Could not find UCDB entry for {city}, {country}")
        return None

    variables = [
        ("GHSL", "GH_POP_TOT", list(range(1975, 2031, 5))),
        ("GHSL", "GH_BUS_TOT", list(range(1975, 2031, 5))),
        ("SOCIOECONOMIC", "SC_GDP_AVG", [1990,1995,2000,2005,2010,2015,2020]),
        ("SOCIOECONOMIC", "SC_SEC_HDI", [1990,1995,2000,2005,2010,2015,2020]),
        ("EXPOSURE", "EX_010_POP", list(range(1975, 2031, 5))),
        ("EMISSIONS", "EM_CO2_TOT", [1975,1990,2000,2005,2010,2015,2020,2022]),
        ("EMISSIONS", "EM_GHG_TOT", [1975,1990,2000,2005,2010,2015,2020,2022]),
        ("EMISSIONS", "EM_NOX_TOT", [1975,1990,2000,2005,2010,2015,2020,2022]),
        ("EMISSIONS", "EM_PM2_TOT", [1975,1990,2000,2005,2010,2015,2020,2022]),
    ]

    # Collect series as list.
    series_list = []
    for sheet_name, prefix, years in variables:
        df = sheets[sheet_name]
        s = extract_ucdb_yearly_series(df, prefix, years, id_col="ID_UC_G0", target_id=uc_id)
        series_list.append(s)

    # Combine into DataFrame.
    ucdb_df = pd.concat(series_list, axis=1)
    # Reindex to full year range and interpolate.
    ucdb_df = ucdb_df.reindex(YEARS)
    ucdb_df = ucdb_df.interpolate(method="linear", limit_direction="both")
    return ucdb_df

def main():
    # Load UCDB sheets.
    xls = pd.ExcelFile(UCDB_FILE)
    sheets = {name: xls.parse(name) for name in xls.sheet_names}

    all_dfs = []
    for filepath in glob.glob(WEATHER_PATTERN):
        # Extract location code from filename (e.g., "17.36N_78.5E.csv").
        location_code = os.path.basename(filepath).replace(".csv", "")
        print(f"Processing {location_code}...")

        # Load weather data for this location.
        weather_df = load_weather_for_location(filepath, location_code)

        # Get UCDB data for this location (if matched).
        ucdb_df = get_ucdb_series_for_location(sheets, location_code)

        if ucdb_df is None:
            print(f"Skipping {location_code}: no UCDB match.")
            continue

        # Merge weather and UCDB on year.
        merged = pd.merge(weather_df, ucdb_df, left_on="year", right_index=True, how="inner")
        # Keep only years 1990-2018 (weather data covers 1980-2018, but UCDB contains some fields that only start in 1990).
        merged = merged[merged["year"].between(1990, 2018)]
        all_dfs.append(merged)

    if not all_dfs:
        print("No matching locations found.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sort_values(["location_code", "year"])

    # Rename columns for clarity.
    rename_map = {
        "GH_POP_TOT": "population",
        "GH_BUS_TOT": "built_up_area_m2",
        "SC_GDP_AVG": "gdp_ppp",
        "SC_SEC_HDI": "hdi",
        "EX_010_POP": "pop_exposed_flood_10yr",
        "EM_CO2_TOT": "co2_emissions_ton",
        "EM_GHG_TOT": "ghg_emissions_ton",
        "EM_NOX_TOT": "nox_emissions_ton",
        "EM_PM2_TOT": "pm25_emissions_ton",
        "temp_mean": "temp_mean_K",
        "wind_mean": "wind_speed_mean_ms",
    }
    final_df = final_df.rename(columns=rename_map)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(final_df)} rows to {OUTPUT_FILE}")
    print(final_df.head())
    print(final_df.groupby("location_code").size())

if __name__ == "__main__":
    main()