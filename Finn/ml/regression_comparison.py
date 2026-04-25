import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})



INPUT_FILE = "datasets/merged_all_locations_yearly_1990_2018.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/regression_comparison/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [
    "co2_emissions_ton",
    "ghg_emissions_ton",
    "nox_emissions_ton",
    "pm25_emissions_ton"
]

WEATHER_FEATURES = [
    "temp_mean_K",
    "wind_speed_mean_ms",
    "precip_total_mm"
]

SOCIO_FEATURES = [
    "population",
    "built_up_area_m2",
    "gdp_ppp",
    "hdi",
    "pop_exposed_flood_10yr"
]

ALL_FEATURES = WEATHER_FEATURES + SOCIO_FEATURES


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
    df = df.sort_values("year")
    return df


def time_split(df, features, target):
    train_df = df[df["year"] <= 2010]
    test_df = df[df["year"] > 2010]

    X_train = train_df[features].values
    X_test = test_df[features].values

    # Log transform target.
    y_train = np.log1p(train_df[target].values)
    y_test = np.log1p(test_df[target].values)

    return X_train, X_test, y_train, y_test


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return r2, rmse, preds


# PLOTS.

def plot_predictions(y_true, y_pred, title, path):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)

    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v])

    plt.xlabel("Actual (log)")
    plt.ylabel("Predicted (log)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_hyderabad_timeseries(df, target, models, features, path):
    city_df = df[df["city"] == "Hyderabad"].sort_values("year")
    years = city_df["year"].values
    y_true = np.log1p(city_df[target].values)
    plt.figure(figsize=(8,5))
    plt.plot(years, y_true, label="Actual", linewidth=2)
    for name, model in models.items():
        model.fit(df[features].values, np.log1p(df[target].values))
        preds = model.predict(city_df[features].values)
        plt.plot(years, preds, linestyle="--", label=name)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Log Emissions", fontsize=12)
    plt.title(f"Hyderabad Time Series - {target}", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# AGGREGATED VISUALISATIONS.

def plot_hyderabad_all_emissions(df, models, features, output_path):
    """Plot actual vs predicted for all four emission targets for Hyderabad."""
    city_df = df[df["city"] == "Hyderabad"].sort_values("year")
    years = city_df["year"].values
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    targets = TARGETS
    axes = axes.flatten()
    
    for idx, target in enumerate(targets):
        y_true = np.log1p(city_df[target].values)
        model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        model.fit(df[features].values, np.log1p(df[target].values))
        y_pred = model.predict(city_df[features].values)
        
        axes[idx].plot(years, y_true, 'o-', label="Actual", linewidth=2)
        axes[idx].plot(years, y_pred, 's--', label="Predicted", linewidth=1.5)
        axes[idx].set_title(target.replace("_", " ").title(), fontsize=14)
        axes[idx].set_xlabel("Year", fontsize=14)
        axes[idx].set_ylabel("Log Emissions", fontsize=14)
        axes[idx].legend(fontsize=12)
        axes[idx].grid(True, linestyle=':', alpha=0.5)
    
    plt.suptitle("Hyderabad: Actual vs Predicted Emissions (Random Forest, Full Features)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_global_heatmap(results_df, output_path):
    """Heatmap of R^2 across targets, models, and feature sets."""
    # Create a combined label for heatmap cells.
    results_df["label"] = results_df["model"] + " (" + results_df["feature_set"] + ")"
    pivot = results_df.pivot(index="target", columns="label", values="r2")
    pivot = pivot.round(3)
    
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.4)
    ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'R²', 'orientation': 'vertical'}, annot_kws={'size': 14})
    # Access the colorbar and set its label to horizontal (not rotated).
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('R²', rotation=0, labelpad=15, fontsize=16)

    ax.set_title("Model Performance Across Targets and Feature Sets", fontsize=16)
    ax.set_xlabel("Model (Feature Set)", fontsize=14)
    ax.set_ylabel("Target Variable", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_model_comparison_bar(results_df, panel_df, output_path):
    """Bar chart comparing average R² for RF, GB, MLP (full features) and Fixed Effects."""
    # Average R^2 across targets for each model (full feature set only).
    rf_full = results_df[(results_df["model"] == "RandomForest") & (results_df["feature_set"] == "full")]["r2"].mean()
    gb_full = results_df[(results_df["model"] == "GradientBoosting") & (results_df["feature_set"] == "full")]["r2"].mean()
    mlp_full = results_df[(results_df["model"] == "MLP") & (results_df["feature_set"] == "full")]["r2"].mean()
    fe_avg = panel_df["r2"].mean()  # Fixed effects already averaged over targets.
    
    categories = ["Random Forest", "Gradient Boosting", "MLP", "Fixed Effects"]
    values = [rf_full, gb_full, mlp_full, fe_avg]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    max_val = max(values)
    ax.set_ylim(0, max_val + 0.15)
    ax.set_ylabel("Average R²")
    ax.set_title("Model Performance Comparison (Full Features)")
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_set_comparison(results_df, output_path):
    """
    Grouped bar chart: average R^2 across all targets for each model (RF, GB, MLP)
    on each feature set (full, weather_only, socio_only).
    """
    # Compute average R^2 across targets for each model and feature set.
    grouped = results_df.groupby(["model", "feature_set"])["r2"].mean().reset_index()
    
    # Pivot for easier plotting.
    pivot = grouped.pivot(index="model", columns="feature_set", values="r2")
    
    # Define order of models.
    models = ["RandomForest", "GradientBoosting", "MLP"]
    pivot = pivot.reindex(models)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pivot.index))
    width = 0.25
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']  # full, weather, socio.
    
    for i, (feat_set, colour) in enumerate(zip(pivot.columns, colours)):
        ax.bar(x + i*width, pivot[feat_set], width, label=feat_set.capitalize(), color=colour)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.set_ylabel("Average R²")
    ax.set_title("Model Performance by Feature Set (averaged across all emission targets)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(df, features, target, output_path):
    """
    Train a Random Forest on the full dataset for a given target (e.g., CO2)
    and plot feature importances as a horizontal bar chart.
    """
    X = df[features].values
    y = np.log1p(df[target].values)
    
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), np.array(features)[indices])
    plt.xlabel("Feature Importance")
    plt.title(f"Random Forest Feature Importance – {target.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    df = load_data()
    results = []

    feature_sets = {
        "full": ALL_FEATURES,
        "weather_only": WEATHER_FEATURES,
        "socio_only": SOCIO_FEATURES
    }

    for target in TARGETS:
        print(f"\n=== Target: {target} ===")

        for set_name, features in feature_sets.items():
            print(f"\nFeature set: {set_name}")

            X_train, X_test, y_train, y_test = time_split(df, features, target)

            models = {
                "RandomForest": RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    random_state=42
                ),

                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                ),

                "MLP": Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        max_iter=5000,
                        early_stopping=True,
                        learning_rate_init=0.001,
                        random_state=42
                    ))
                ])
            }

            for name, model in models.items():
                print(f"Training {name}...")

                r2, rmse, preds = evaluate(
                    model, X_train, X_test, y_train, y_test
                )

                print(f"R2: {r2:.3f} | RMSE: {rmse:.3f}")

                results.append({
                    "target": target,
                    "model": name,
                    "feature_set": set_name,
                    "r2": r2,
                    "rmse": rmse
                })

                plot_predictions(
                    y_test,
                    preds,
                    f"{name} ({set_name}) - {target}",
                    f"{OUTPUT_DIR}/{target}_{name}_{set_name}.png"
                )

                # Only do once per target using full features (avoids clutter).
                if set_name == "full":
                    plot_hyderabad_timeseries(
                        df,
                        target,
                        models,
                        features,
                        f"{OUTPUT_DIR}/{target}_hyderabad_timeseries.png"
                    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)


    # GLOBAL VISUALISATIONS.

    # Heatmap (model vs feature set per target).
    for target in TARGETS:
        sub = results_df[results_df["target"] == target]

        pivot = sub.pivot(index="model", columns="feature_set", values="r2")

        plt.figure(figsize=(6,4))
        sns.heatmap(pivot, annot=True, fmt=".2f")
        plt.title(f"Performance Heatmap - {target}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{target}_heatmap.png", dpi=300)
        plt.close()


    # Feature set comparison (averaged).
    avg = results_df.groupby("feature_set")["r2"].mean()

    plt.figure(figsize=(5,4))
    avg.plot(kind="bar")
    plt.ylabel("Average R²")
    plt.title("Feature Set Comparison (All Targets)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_set_comparison.png", dpi=300)
    plt.close()


    # Model comparison (averaged).
    avg_model = results_df.groupby("model")["r2"].mean()

    plt.figure(figsize=(5,4))
    avg_model.plot(kind="bar")
    plt.ylabel("Average R²")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=300)
    plt.close()

    # Retrieve panel regression results for comparison.
    panel_df = pd.read_csv("DataLakers/Finn/ml/outputs/panel_model/panel_results.csv")

    # Aggregated plots.
    plot_hyderabad_all_emissions(df, models, ALL_FEATURES, f"{OUTPUT_DIR}/hyderabad_all_emissions_timeseries.png")
    plot_global_heatmap(results_df, f"{OUTPUT_DIR}/global_performance_heatmap.png")
    plot_model_comparison_bar(results_df, panel_df, f"{OUTPUT_DIR}/model_comparison_with_fe.png")

    plot_feature_set_comparison(results_df, f"{OUTPUT_DIR}/feature_set_comparison_grouped.png")
    # Choose one target to show feature importance (e.g., CO2).
    plot_feature_importance(df, ALL_FEATURES, "co2_emissions_ton", 
                            f"{OUTPUT_DIR}/feature_importance_co2.png")

    print("\nDone.")


if __name__ == "__main__":
    main()