import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

UCDB_FILE = "datasets/ghs_ucdb/GHS_UCDB_GLOBE_R2024A.xlsx"
WEATHER_PATTERN = "datasets/weatherbench/*.csv"
OUTPUT_FILE = "datasets/merged_hyderabad_yearly_1990_2018.csv"
TARGET_CITY = "Hyderabad"
TARGET_COUNTRY = "India"
YEARS = list(range(1990, 2019))  # 1990-2018.

def extract_ucdb_yearly_series(df_sheet, prefix, years, id_col="ID_UC_G0", target_id=None):
    series = {}
    for year in years:
        col = f"{prefix}_{year}"
        if col in df_sheet.columns:
            val = df_sheet.loc[df_sheet[id_col] == target_id, col].values
            series[year] = val[0] if len(val) > 0 else np.nan
    return series

def load_weather_hyderabad():
    hyderabad_file = None
    for f in glob.glob(WEATHER_PATTERN):
        if "17.36N_78.5E" in f:
            hyderabad_file = f
            break
    if hyderabad_file is None:
        raise ValueError("Hyderabad weather file not found.")
    
    df = pd.read_csv(hyderabad_file, header=None, names=["year","month","day","hour","temperature","precipitation","u_wind","v_wind"])
    df["wind_speed"] = (df["u_wind"]**2 + df["v_wind"]**2)**0.5

    yearly = df.groupby("year").agg(
        temp_mean=("temperature", "mean"),
        precip_total=("precipitation", "sum"),
        wind_mean=("wind_speed", "mean")
    ).reset_index()
    yearly["precip_mm"] = yearly["precip_total"] * 1000
    yearly = yearly.rename(columns={"precip_total": "precip_m"})

    return yearly


def main():
    xls = pd.ExcelFile(UCDB_FILE)
    sheets = {name: xls.parse(name) for name in xls.sheet_names}

    general = sheets["GENERAL_CHARACTERISTICS"]
    hyderabad_row = general[(general["GC_UCN_MAI_2025"] == TARGET_CITY) & (general["GC_CNT_GAD_2025"] == TARGET_COUNTRY)]
    if hyderabad_row.empty:
        print("Hyderabad not found.")
        return
    
    hyderabad_id = hyderabad_row["ID_UC_G0"].iloc[0]
    print(f"Found Hyderabad with ID {hyderabad_id}")

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

    data = {}
    for sheet_name, prefix, years in variables:
        df = sheets[sheet_name]
        series = extract_ucdb_yearly_series(df, prefix, years, id_col="ID_UC_G0", target_id=hyderabad_id)
        for year, val in series.items():
            if year not in data:
                data[year] = {}
            data[year][prefix] = val

    ucdb_df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    # Reindex to full year range (1990-2018) and interpolate.
    ucdb_df = ucdb_df.reindex(YEARS)
    ucdb_df = ucdb_df.interpolate(method="linear", limit_direction="both")

    weather_df = load_weather_hyderabad()
    weather_df["year"] = weather_df["year"].astype(int)

    merged = pd.merge(weather_df, ucdb_df, left_on="year", right_index=True, how="inner")
    merged = merged.set_index("year").sort_index()

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
        "precip_mm": "precip_total_mm",
        "wind_mean": "wind_speed_mean_ms",
    }
    merged = merged.rename(columns=rename_map)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE)
    print(f"Saved {len(merged)} rows to {OUTPUT_FILE}")
    print(merged.head())
    print(merged.info())

if __name__ == "__main__":
    main()