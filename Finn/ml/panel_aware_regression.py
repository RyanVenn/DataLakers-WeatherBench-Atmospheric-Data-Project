import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


INPUT_FILE = "datasets/merged_all_locations_yearly_1990_2018.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/panel_model/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [
    "co2_emissions_ton",
    "ghg_emissions_ton",
    "nox_emissions_ton",
    "pm25_emissions_ton"
]

FEATURES = [
    "temp_mean_K",
    "wind_speed_mean_ms",
    "precip_total_mm",
    "population",
    "built_up_area_m2",
    "gdp_ppp",
    "hdi",
    "pop_exposed_flood_10yr"
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

    df = df.dropna()
    df = df.sort_values(["city", "year"])

    return df


def apply_fixed_effects(df, features, target):
    df_fe = df.copy()

    # Log target.
    df_fe[target] = np.log1p(df_fe[target])

    # Subtract city mean (within transform); this is called demeaning.
    for col in features + [target]:
        df_fe[col] = df_fe[col] - df_fe.groupby("city")[col].transform("mean")

    return df_fe


def time_split(df, features, target):
    train = df[df["year"] <= 2010]
    test = df[df["year"] > 2010]

    X_train = train[features].values
    X_test = test[features].values

    y_train = train[target].values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return r2, rmse, preds



def plot_predictions(y_true, y_pred, title, path):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)

    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v])

    plt.xlabel("Actual (log, demeaned)")
    plt.ylabel("Predicted")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()



def main():
    df = load_data()
    results = []

    for target in TARGETS:
        print(f"\n=== Target: {target} ===")

        df_fe = apply_fixed_effects(df, FEATURES, target)
        X_train, X_test, y_train, y_test = time_split(df_fe, FEATURES, target)

        model = LinearRegression()

        r2, rmse, preds = evaluate(model, X_train, X_test, y_train, y_test)
        print(f"Fixed Effects Model: R2: {r2:.3f}, RMSE: {rmse:.3f}")
        results.append({
            "target": target,
            "model": "fixed_effects",
            "r2": r2,
            "rmse": rmse
        })

        plot_predictions(
            y_test,
            preds,
            f"Fixed Effects - {target}",
            f"{OUTPUT_DIR}/{target}_fixed_effects.png"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/panel_results.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()