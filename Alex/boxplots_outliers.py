import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
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
    columns = ["temp_mean_K", "precip_total_mm", "wind_speed_mean_ms"]
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

    # Filter once for the city
    city_df = df[df["city"] == city]

    # Extract columns directly as arrays (no loops needed)
    temps = city_df["temp_mean_K"].dropna().values
    precips = city_df["precip_total_mm"].dropna().values
    winds = city_df["wind_speed_mean_ms"].dropna().values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Temperature plot
    axes[0].boxplot(temps)
    axes[0].set_title("Temperature (K)")
    axes[0].set_xlabel("")
    axes[0].grid(True)

    # Precipitation plot
    axes[1].boxplot(precips)
    axes[1].set_title("Precipitation (mm)")
    axes[1].set_xlabel("")
    axes[1].grid(True)

    # Wind plot
    axes[2].boxplot(winds)
    axes[2].set_title("Wind Speed (m/s)")
    axes[2].set_xlabel("")
    axes[2].grid(True)

    # get rid of the value 1 on x-axis
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xticklabels([])

    # output outliers in easy to read format all in one place
    temp_outliers = get_outliers(temps)
    precip_outliers = get_outliers(precips)
    wind_outliers = get_outliers(winds)
    print("Outliers for,", city ,"temp:", temp_outliers, "precip:", precip_outliers, "wind:", wind_outliers)

    # formatting and png save
    fig.suptitle(f"Weather Distribution for {city}")
    plt.tight_layout()
    plt.savefig("boxplot "+city+".png", dpi=300, bbox_inches="tight")

    # stops the plots appearing when all is selected 
    if not all:
        plt.show()


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




