import pandas as pd
import glob
import os


COLUMN_NAMES = [
    "year",
    "month",
    "day",
    "hour",
    "temperature",
    "precipitation",
    "u_wind",
    "v_wind",
]


LOCATION_MAP = {
    "-33.5N_151E": "Maroota (Australia)",
    "-33.9N_18.5E": "Cape Town (South Africa)",
    "17.36N_78.5E": "Hyderabad (India)",
    "18N_283.2E": "Kingston (Jamaica)",
    "40.75N_286.01E": "New York (USA)",
    "41.9N_12.46E": "Rome (Italy)",
    "43.28N_5.39E": "Marseille (France)",
    "43.64N_280.63E": "Toronto (Canada)",
    "51.03N_245.94E": "Calgary (Canada)",
    "51.5N_359.9E": "London (UK)",
    "58.76N_265.83E": "Churchill (Canada)",
    "59.92N_10.75E": "Oslo (Norway)",
}


def load_weather_data(path_pattern: str) -> pd.DataFrame:
    """
    Loads all CSV files matching a wildcard path and combines them into a single dataframe.
    Adds a 'location' column derived from the filename.
    """

    files = glob.glob(path_pattern)
    if not files:
        raise ValueError("No CSV files found for pattern: " + path_pattern)

    dfs = [] # Store dataframes for each file.
    for file in files:
        location_code = os.path.basename(file).replace(".csv", "")
        location_name = LOCATION_MAP.get(location_code, location_code)

        df = pd.read_csv(
            file,
            header=None,
            names=COLUMN_NAMES
        )
        df["location_code"] = location_code
        df["location_name"] = location_name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Construct timestamp.
    combined["timestamp"] = pd.to_datetime(
        combined[["year", "month", "day", "hour"]]
    )

    # Add a wind speed feature.
    combined["wind_speed"] = (combined["u_wind"]**2 + combined["v_wind"]**2) ** 0.5

    combined["precipitation_mm"] = combined["precipitation"] * 1000

    return combined