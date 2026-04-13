import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

##How to run: 
# python boxplot_outliers ...
# --city [pick a city from the list or 'all' to do all cities]
# --clean               
# clean is optional, use to make the cleaned dataset (can be for specific city or all)


DATASET = "./merged_all_locations_yearly_1990_2018.csv"


parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, required=False, help="City name")
parser.add_argument("--clean", action="store_true", help="Remove outliers")


args = parser.parse_args()

city = args.city
clean = args.clean

cities = ["Hyderabad", "Marseille", "Rome", "Kingston", "London", "Toronto", "Calgary", "Oslo", "Cape Town", "New York City"]


# calculates outliers for a given array
def get_outliers(values):
    arr = np.array(values, dtype=float)

    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = arr[(arr < lower) | (arr > upper)]
    return outliers

# removes outliers based on city: i.e. removes what is considered outlier for given city, for all cities, rather than what is considered outlier across all cities 
def remove_outliers(df):
    columns = ["temp_mean_K", "precip_total_mm", "wind_speed_mean_ms", "population", "built_up_area_m2", "gdp_ppp", "hdi", "pop_exposed_flood_10yr", "co2_emissions_ton", "ghg_emissions_ton", "nox_emissions_ton", "pm25_emissions_ton"]
    keep_mask = pd.Series(True, index=df.index)

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        col_keep = (df[column] >= lower) & (df[column] <= upper)

        outliers = df[~col_keep]
        if not outliers.empty:
            print(f"\nOutliers in {column}:")
            print(outliers[["year", "city", column]])

        keep_mask &= col_keep

    return df[keep_mask].copy()

# removes outliers for particular dataset
def remove_outliers_and_save():
    df = pd.read_csv(DATASET)


    # given one city, a city-specific dataset is made
    if city is not None and city != "all":
        city_df = df[df["city"] == city].copy()
        cleaned_df = remove_outliers(city_df)

        cleaned_df = cleaned_df.drop(columns=["city"])
        formatted = city.lower().replace(" ", "_")
        output_path = f"merged_{formatted}_yearly_1990_2018_clean.csv"

        print(f"\nSaved cleaned dataset to: {output_path}")
        print(f"Rows before: {len(city_df)}")
        print(f"Rows after: {len(cleaned_df)}")

    # when all or none is selected, do cleaning for all cities
    else:
        cleaned_parts = []

        for city_name in df["city"].dropna().unique():
            city_df = df[df["city"] == city_name].copy()
            cleaned_city_df = remove_outliers(city_df)
            cleaned_parts.append(cleaned_city_df)

        cleaned_df = pd.concat(cleaned_parts, ignore_index=True)

        base, ext = os.path.splitext(DATASET)
        output_path = f"{base}_clean{ext}"

        print(f"\nSaved cleaned dataset to: {output_path}")
        print(f"Rows before: {len(df)}")
        print(f"Rows after: {len(cleaned_df)}")

    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df


def draw_avg_temp_boxplot(city, all):
    df = pd.read_csv(DATASET)

    # filter for the city
    city_df = df[df["city"] == city]

    features = {
        "Temperature (K)": city_df["temp_mean_K"].dropna().values,
        "Precipitation (mm)": city_df["precip_total_mm"].dropna().values,
        "Wind Speed (m/s)": city_df["wind_speed_mean_ms"].dropna().values,
        "Population": city_df["population"].dropna().values,
        "HDI": city_df["hdi"].dropna().values,
        "Built-up Area (m²)": city_df["built_up_area_m2"].dropna().values,
        "GDP PPP": city_df["gdp_ppp"].dropna().values,
        "Flood Exposed Population": city_df["pop_exposed_flood_10yr"].dropna().values,
        "CO2 Emissions (ton)": city_df["co2_emissions_ton"].dropna().values,
        "GHG Emissions (ton)": city_df["ghg_emissions_ton"].dropna().values,
        "NOx Emissions (ton)": city_df["nox_emissions_ton"].dropna().values,
        "PM2.5 Emissions (ton)": city_df["pm25_emissions_ton"].dropna().values,
    }

    # compute outliers for all features
    all_outliers = {}
    for name, values in features.items():
        all_outliers[name] = get_outliers(values)

    print(f"Outliers for {city}:")
    for feature, outliers in all_outliers.items():
        print(f"  {feature}: {outliers}")

    # only keep non-empty arrays for plotting
    plot_features = {name: values for name, values in features.items() if len(values) > 0}

    n_features = len(plot_features)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 7 * n_rows))
    axes = axes.flatten()

    fig.subplots_adjust(wspace=0.4)

    for i, (name, values) in enumerate(plot_features.items()):
        axes[i].boxplot(values)
        axes[i].set_title(name, fontsize=18)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].set_xticklabels([])
        axes[i].grid(True)

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")


    plt.savefig(f"boxplot_{city}.png", dpi=300, bbox_inches="tight")

    if not all:
        plt.show()
    else:
        plt.close()


# handles all cities being selected vs one
if city is not None: 
    if city == "all":
        for x in range(0, len(cities)):
            draw_avg_temp_boxplot(cities[x], True)
    else:
        draw_avg_temp_boxplot(city, False)

# handles cleaning flag
if clean:
    remove_outliers_and_save()




