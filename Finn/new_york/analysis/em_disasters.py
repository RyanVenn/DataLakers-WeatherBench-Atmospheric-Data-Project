import pandas as pd
import matplotlib.pyplot as plt

FILE = "datasets/em_dat/united_states.xlsx"

df = pd.read_excel(FILE)

print("Countries:", df["Country"].unique())
print("Disaster Groups:", df["Disaster Group"].unique())
print("Disaster Types:", df["Disaster Type"].unique())

df = df[df["Country"].str.contains("United States", na=False)]

# Natural disasters.
df = df[df["Disaster Group"] == "Natural"]

# Convert year.
df["year"] = pd.to_numeric(df["Start Year"], errors="coerce")
df = df.dropna(subset=["year"])

print("After basic filters:", len(df))

yearly = df.groupby(["year", "Disaster Type"]).size().unstack(fill_value=0)

print("Final shape:", yearly.shape)

smooth = yearly.rolling(5, min_periods=1).mean()

plt.figure(figsize=(12,6))

# Only plot top 5 most common types (prevents clutter)
top_types = yearly.sum().sort_values(ascending=False).head(5).index

for col in top_types:
    plt.plot(yearly.index, yearly[col], alpha=0.3)
    plt.plot(smooth.index, smooth[col], label=col)

plt.legend()
plt.title("Disaster Frequency in USA (EM-DAT)")
plt.xlabel("Year")
plt.ylabel("Number of Events")

plt.grid(alpha=0.2)
plt.show()