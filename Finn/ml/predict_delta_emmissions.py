import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


INPUT_FILE = "datasets/merged_all_locations_yearly_1990_2018.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/delta_emmissions"
os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURES = [
    "temp_mean_K",
    "wind_speed_mean_ms",
    "precip_total_mm",
    "population",
    "built_up_area_m2",
    "gdp_ppp",
    "hdi"
]

TARGET = "co2_emissions_ton"


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

    # Target: year-to-year change.
    df["target_change"] = df.groupby("location_code")[TARGET].diff()

    # Lag features.
    for col in FEATURES:
        df[f"{col}_lag1"] = df.groupby("location_code")[col].shift(1)

    df = df.dropna()

    return df


def plot_pred_vs_actual(y_test, y_pred, name):
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual change")
    plt.ylabel("Predicted change")
    plt.title(name + " - Predicted vs Actual")

    plt.savefig(f"{OUTPUT_DIR}/{name}_pred_vs_actual.png")
    plt.close()


def plot_residuals(y_test, y_pred, name):
    residuals = y_test - y_pred

    plt.figure()
    plt.hist(residuals, bins=50)
    plt.title(name + " - Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Count")

    plt.savefig(f"{OUTPUT_DIR}/{name}_residuals.png")
    plt.close()


def plot_timeseries(df, model, scaler, features, name):
    # Choose a few cities to visualise.
    sample_cities = df["city"].unique()[:3]

    for city in sample_cities:
        subset = df[df["city"] == city]

        X = scaler.transform(subset[features])
        y_true = subset["target_change"].values
        y_pred = model.predict(X)

        plt.figure()
        plt.plot(subset["year"], y_true, label="Actual")
        plt.plot(subset["year"], y_pred, label="Predicted")

        plt.title(f"{name} - {city}")
        plt.xlabel("Year")
        plt.ylabel("Change in CO2")

        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/{name}_{city}_timeseries.png")
        plt.close()


def train_models(X_train, X_test, y_train, y_test, df, scaler, features):
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "SVR": SVR(C=1.0, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "GBR": GradientBoostingRegressor()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        results[name] = mae

        # Plots.
        plot_pred_vs_actual(y_test, pred, name)
        plot_residuals(y_test, pred, name)
        plot_timeseries(df, model, scaler, features, name)

    return results


def main():
    df = load_data()
    features = [f"{col}_lag1" for col in FEATURES]
    X = df[features].values
    y = df["target_change"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = train_models(X_train, X_test, y_train, y_test, df, scaler, features)
    print("\nModel Comparison (MAE):")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    baseline_pred = np.zeros_like(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"Baseline (no change): {baseline_mae:.4f}")


if __name__ == "__main__":
    main()