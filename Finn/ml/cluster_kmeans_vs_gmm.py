import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


INPUT_FILE = "datasets/merged_all_locations_yearly_1990_2018.csv"
OUTPUT_DIR = "DataLakers/Finn/ml/outputs/clustering_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURES = [
    "temp_mean_K",
    "wind_speed_mean_ms",
    "precip_total_mm",
    "population",
    "built_up_area_m2",
    "gdp_ppp",
    "hdi",
    "pop_exposed_flood_10yr",
    "co2_emissions_ton",
    "ghg_emissions_ton",
    "nox_emissions_ton",
    "pm25_emissions_ton"
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
    return df


def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    return model, labels


def run_gmm(X, k):
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(X)
    probs = model.predict_proba(X)
    return model, labels, probs


def main():
    df = load_data()

    X = df[FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    for k in range(2, 26):
        # KMeans.
        km_model, km_labels = run_kmeans(X_scaled, k)
        km_score = silhouette_score(X_scaled, km_labels)

        # GMM.
        gmm_model, gmm_labels, gmm_probs = run_gmm(X_scaled, k)
        gmm_score = silhouette_score(X_scaled, gmm_labels)

        bic = gmm_model.bic(X_scaled)
        aic = gmm_model.aic(X_scaled)

        results.append({
            "k": k,
            "kmeans_silhouette": km_score,
            "gmm_silhouette": gmm_score,
            "gmm_bic": bic,
            "gmm_aic": aic
        })

        print(f"k={k} | KM silhouette={km_score:.3f} | GMM silhouette={gmm_score:.3f} | BIC={bic:.1f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

    # Use best k (manual or choose lowest BIC; in this case we hardcode 10 as it has the highest silhouette score with a low BIC).
    # best_k = results_df.sort_values("gmm_bic").iloc[0]["k"]
    best_k = 10
    best_k = int(best_k)

    print(f"\nBest k (by GMM BIC): {best_k}")

    # Final models.
    km_model, km_labels = run_kmeans(X_scaled, best_k)
    gmm_model, gmm_labels, gmm_probs = run_gmm(X_scaled, best_k)

    df["kmeans_cluster"] = km_labels
    df["gmm_cluster"] = gmm_labels

    # Save clustered data (_10 denotes the hardcoding of best_k = 10).
    df.to_csv(f"{OUTPUT_DIR}/clustered_data_10.csv", index=False)

    # Save GMM uncertainty.
    uncertainty = 1 - gmm_probs.max(axis=1)
    df["gmm_uncertainty"] = uncertainty

    df.to_csv(f"{OUTPUT_DIR}/clustered_with_uncertainty.csv", index=False)

    print("Saved clustering outputs.")


if __name__ == "__main__":
    main()