import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import glob
from pathlib import Path

columns = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Temperature",
    "Percipitation",
    "U-wind",
    "V-wind"
]

locations = {
    "-33.5N_151E": "Australia",
    "-33.9N_18.5E": "South Africa",
    "17.36N_78.5E": "India",
    "18N_283.2E": "Jamaica",
    "40.75N_286.01E": "USA",
    "41.9N_12.46E": "Italy",
    "43.28N_5.39E": "France",
    "43.64N_280.63E": "Canada (Toronto)", 
    "51.03N_245.94E": "Canada (Calgary)",
    "51.5N_359.9E": "United Kingdom",
    "58.76N_265.83E": "Canada (Churchill)",
    "59.92N_10.75E": "Norway"
}

files = glob.glob("dataset/*.csv")
dfs = []

for file in files:
    df = pd.read_csv(file, header=None, names=columns)
    df["location"] = file.split("/")[-1].replace(".csv", "")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data["datetime"] = pd.to_datetime(data[["Year", "Month", "Day", "Hour"]])


# ----- BOX & WHISKER PLOTS -----
# plot of all four atmospheric variables 
fig = plt.figure(figsize = (10, 7))
plt.boxplot([
    data["Temperature"],
    data["Percipitation"],
    data["U-wind"],
    data["V-wind"]
])

plt.xticks([1, 2, 3, 4], ["Temperature", "Percipitation", "U-wind", "V-wind"])
plt.title("Distribution of Variables")
plt.show()

# plot for just percipitation
log_percipitation = np.log(data["Percipitation"] + 1e-8)

fig = plt.figure(figsize = (10, 7))
plt.boxplot([
    log_percipitation
])

plt.xticks([1], ["Percipitation"])
plt.title("Distribution of Percipitation Values")
plt.show()

# plot for U-wind and V-wind 
fig = plt.figure(figsize = (10, 7))
plt.boxplot([
    data["U-wind"],
    data["V-wind"]
])

plt.xticks([1, 2], ["U-wind", "V-wind"])
plt.title("Distribution of U-wind and V-wind")
plt.show()


# ----- TIME SERIES PLOTS ------
# plot for long term average temperature change
yearly_temperature = data.groupby("Year")["Temperature"].mean()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(yearly_temperature.index, yearly_temperature.values, linewidth=3)
ax.set_title("Average Temperature Over Time (All Locations)")
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (K)')
ax.grid()
plt.tight_layout()
plt.show()

# average temperature per locations
data["Temperature_C"] = data["Temperature"] - 273.15
data["country"] = data["location"].map(locations)

fig, ax = plt.subplots(figsize=(12,7))
countries = data["country"].unique()
colours = plt.cm.tab20(np.linspace(0, 1, len(countries)))

for (country, group), colour in zip(data.groupby("country"), colours):
    yearly_temp = group.groupby("Year")["Temperature_C"].mean()
    
    ax.plot(
        yearly_temp.index,
        yearly_temp.values,
        linewidth=2,
        label=country,
        color=colour
    )

ax.set_title("Average Temperature Trend by Location (1980–2018)")
ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.grid(True)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# precipitation over time 
yearly_precipitation = data.groupby("Year")["Percipitation"].mean()

fig, ax = plt.subplots(figsize=(12,7))
ax.plot(yearly_precipitation.index, yearly_precipitation.values, linewidth=3)
ax.set_title("Average Precipitation Over Time (All Locations)")
ax.set_xlabel('Year')
ax.set_ylabel('Precipitation (m)')
ax.grid()
plt.tight_layout()
plt.show()


# ----- PRECIPITATION & WIND IMPACT ON TEMPERATURE -----
# relationship between temperature and precipitation over time
yearly_data = data.groupby("Year").agg({
    "Temperature_C": "mean",
    "Percipitation": "mean"
})

plt.figure(figsize=(8,6))
plt.scatter(
    yearly_data["Percipitation"],
    yearly_data["Temperature_C"],
    alpha=0.7
)
plt.xlabel("Average Precipitation")
plt.ylabel("Average Temperature")
plt.title("Relationship between Precipitation and Temperature")
plt.grid(True)
plt.show()
print(yearly_data.corr())

# relationship between temperature and wind over time
data["wind_speed"] = np.sqrt(data["U-wind"]**2 + data["V-wind"]**2)

yearly_data_2 = data.groupby("Year").agg({
    "Temperature_C": "mean",
    "wind_speed": "mean"
})

plt.figure(figsize=(8,6))
plt.scatter(
    yearly_data_2["wind_speed"],
    yearly_data_2["Temperature_C"],
    alpha=0.7
)
plt.xlabel("Average Wind Speed")
plt.ylabel("Average Temperature (Celsius)")
plt.title("Relationship between Wind Speed and Temperature")
plt.grid(True)
plt.show()

print(yearly_data_2.corr())


# ----- CORRELATION HEATMAP -----
corr_data = data[[
    "Temperature_C",
    "Percipitation",
    "U-wind",
    'V-wind',
    'wind_speed'
]]

co_matrix = corr_data.corr()
print(co_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(co_matrix, annot=True)
plt.title("Correlation Matrix of Atmospheric Variables")
plt.show()


# ----- SEASONAL TEMPERATURE PATTERN -----
monthly_temp = data.groupby("Month")["Temperature_C"].mean()

plt.figure(figsize=(8,6))

plt.plot(
    monthly_temp.index,
    monthly_temp.values,
    marker="o",
    linewidth = 2
)

plt.title("Average Temperature by Month")
plt.xlabel("Month")
plt.ylabel("Average Temperature (Celsius)")
plt.grid(True)
plt.show()


# ----- CORRELATION BETWEEN SPECIFIC LOCATIONS, PRECIPITATION AND WIND -----
location_key = "43.64N_280.63E"
location_name = locations[location_key]

canada_data = data[data["location"] == location_key] 

canada_correlation = canada_data.groupby("Year").agg({
    "Temperature_C": "mean",
    "Percipitation": "mean"
})

plt.figure(figsize=(8,6))
x = canada_correlation["Percipitation"]
y = canada_correlation["Temperature_C"]

plt.scatter(x, y, alpha = 0.7)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

plt.xlabel("Average Precipitation")
plt.ylabel("Average Temperature")
plt.title("Temperature vs Precipitation: Canada")
plt.grid(True)
plt.show()

# ----- WIND CORRELATION CANADA -----
canada_wind_correlation = canada_data.groupby("Year").agg({
    "Temperature_C": "mean",
    "wind_speed": "mean"
})

plt.figure(figsize=(8,6))
x_2 = canada_wind_correlation["wind_speed"]
y_2 = canada_wind_correlation["Temperature_C"]

plt.scatter(x_2, y_2, alpha=0.7)

m, b = np.polyfit(x_2, y_2, 1)
plt.plot(x_2, m*x_2 + b)

plt.xlabel("Average Wind Speed")
plt.ylabel("Average Temperature")
plt.title("Temperature vs Wind Speed: Canada")
plt.grid(True)
plt.show()

