import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os


INPUT_FILE = "datasets/merged_all_locations_yearly_1990_2018.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/transition_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURES = [
    "temp_mean_K",
    "wind_speed_mean_ms",
    "precip_total_mm",
    "population",
    "built_up_area_m2",
    "co2_emissions_ton"
]


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df.columns = [
        "year","temp_mean_K","wind_speed_mean_ms","precip_total_mm",
        "location_code","city","country",
        "population","built_up_area_m2","gdp_ppp","hdi",
        "pop_exposed_flood_10yr",
        "co2_emissions_ton","ghg_emissions_ton",
        "nox_emissions_ton","pm25_emissions_ton"
    ]

    df = df.sort_values(["location_code", "year"])
    df = df.dropna()

    return df


def main():
    df = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    # Cluster first
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Create next-year target
    df["next_cluster"] = df.groupby("location_code")["cluster"].shift(-1)

    df = df.dropna(subset=["next_cluster"])

    X = df[FEATURES]
    y = df["next_cluster"]

    # Train/test split (time-based)
    train = df[df["year"] <= 2010]
    test = df[df["year"] > 2010]

    X_train = train[FEATURES]
    y_train = train["next_cluster"]

    X_test = test[FEATURES]
    y_test = test["next_cluster"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Next-state prediction accuracy: {acc:.3f}")

    df.to_csv(f"{OUTPUT_DIR}/cluster_transitions.csv", index=False)


if __name__ == "__main__":
    main()