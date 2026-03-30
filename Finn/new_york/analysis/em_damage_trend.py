import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILE = "datasets/em_dat/united_states.xlsx"

df = pd.read_excel(FILE)

df["year"] = pd.to_numeric(df["Start Year"], errors="coerce")
df["damage"] = pd.to_numeric(df["Total Damage ('000 US$)"], errors="coerce")

df = df.dropna(subset=["year"])
df = df[df["Disaster Group"] == "Natural"]

# Focus on relevant types.
df = df[df["Disaster Type"].isin(["Flood", "Storm", "Earthquake", "Drought", "Wildfire"])]

# Aggregate damage by year.
damage_yearly = df.groupby("year")["damage"].sum()

trend = damage_yearly.rolling(5, min_periods=1).mean()

plt.figure(figsize=(10,6))

plt.plot(trend.index, trend.values)

plt.title("NYC Disaster Damage Trend")
plt.xlabel("Year")
plt.ylabel("Total Damage ('000 USD)")

Path("data_visualisation/new_york/visualisations").mkdir(exist_ok=True)
plt.savefig("data_visualisation/new_york/visualisations/ny_damage_trend.png", dpi=300)
plt.close()