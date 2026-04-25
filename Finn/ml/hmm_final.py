import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})


INPUT_FILE = "datasets/merged_all_locations_yearly_1990_2018.csv"
BASE_OUTPUT = "DataLakers/Finn/ml/outputs/hmm_final/"
os.makedirs(BASE_OUTPUT, exist_ok=True)

ALL_FEATURES = [
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

STATE_RANGE = range(2, 5)


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
    df["year"] = df["year"].astype(int)

    return df


def compute_bic(model, X):
    logL = model.score(X)
    n_samples, n_features = X.shape
    n_states = model.n_components

    n_params = (
        n_states * n_features * 2 +
        n_states * (n_states - 1) +
        (n_states - 1)
    )

    return -2 * logL + n_params * np.log(n_samples)


# Select the best model based on BIC and then refit it to get the best state sequence.
def fit_best_hmm(X):
    best_bic = np.inf
    best_k = None

    for k in STATE_RANGE:
        for seed in range(5):
            try:
                model = GaussianHMM(
                    n_components=k,
                    covariance_type="diag",
                    n_iter=300,
                    min_covar=1e-3,
                    random_state=seed
                )

                model.fit(X)
                bic = compute_bic(model, X)

                if bic < best_bic:
                    best_bic = bic
                    best_k = k

            except:
                continue

    return best_k, best_bic

def fit_final_model(X, k):
    best_model = None
    best_score = -np.inf

    for seed in range(10):
        try:
            model = GaussianHMM(
                n_components=k,
                covariance_type="diag",
                n_iter=300,
                min_covar=1e-3,
                random_state=seed
            )

            model.fit(X)
            score = model.score(X)

            if score > best_score:
                best_score = score
                best_model = model

        except:
            continue

    return best_model


def compute_transition_metrics(states):
    states = np.array(states)

    # Number of transitions.
    transitions = np.sum(states[1:] != states[:-1])

    # State Durations.
    durations = []
    current_len = 1

    for i in range(1, len(states)):
        if states[i] == states[i-1]:
            current_len += 1
        else:
            durations.append(current_len)
            current_len = 1

    durations.append(current_len)
    avg_duration = np.mean(durations)

    return transitions, avg_duration

def reorder_states_by_mean(X_scaled, states, model):
    """
    Reorder state labels so that state 1 has the smallest mean (first PC),
    state 2 the next, etc. Returns a new state sequence and a permuted transition matrix.
    """
    # Compute mean of the first principal component for each state.
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).flatten()
    state_means = {s: pc1[states == s].mean() for s in np.unique(states)}
    # Sort states by mean.
    sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])
    # Create mapping: old_state-new_state (starting from 0 or 1).
    mapping = {old: i for i, old in enumerate(sorted_states)}
    # Apply mapping to state sequence.
    new_states = np.array([mapping[s] for s in states])
    # Permute rows/columns of transition matrix.
    n_states = model.n_components
    perm = np.array([mapping[i] for i in range(n_states)])
    new_transmat = model.transmat_[perm][:, perm]
    # Update model.
    model.transmat_ = new_transmat
    return new_states, model


def plot_state_timeline(city, years, states, outdir):
    plt.figure(figsize=(10, 3))
    plt.plot(years, states, marker='o')
    plt.yticks(sorted(np.unique(states)))
    plt.xlabel("Year")
    plt.ylabel("State")
    plt.title(f"{city} - HMM State Timeline")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{city}_timeline.png", dpi=300)
    plt.close()

def plot_transition_matrix(city, matrix, outdir):
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, aspect='auto')
    plt.colorbar()
    plt.title(f"{city} - Transition Matrix")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{city}_matrix.png", dpi=300)
    plt.close()



def run_hmm(df, features, label):
    print(f"\n=== Running HMM: {label} ===")

    outdir = f"{BASE_OUTPUT}/{label}"
    os.makedirs(outdir, exist_ok=True)

    results = []
    for city, group in df.groupby("city"):
        group = group.sort_values("year")
        X = group[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA only if multi-feature.
        if len(features) > 1:
            X_scaled = PCA(n_components=2).fit_transform(X_scaled)

        k, bic = fit_best_hmm(X_scaled)
        model = fit_final_model(X_scaled, k)

        if model is None:
            continue

        states = model.predict(X_scaled)
        # Reorder states so that state numbers increase with mean PC1 for better interpretability (otherwise HMM labels are arbitrary).
        states, model = reorder_states_by_mean(X_scaled, states, model)

        # Metrics.
        transitions, avg_duration = compute_transition_metrics(states)
        persistence = np.mean(np.diag(model.transmat_))

        # Save visuals.
        plot_state_timeline(city, group["year"].values, states, outdir)
        plot_transition_matrix(city, model.transmat_, outdir)

        results.append({
            "city": city,
            "n_states": k,
            "bic": bic,
            "transitions": transitions,
            "avg_state_duration": avg_duration,
            "self_transition_prob": persistence
        })

        print(f"{city}: states={k}, transitions={transitions}, duration={avg_duration:.2f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(f"{outdir}/summary.csv", index=False)
    return df_out



def main():
    df = load_data()

    # Full feature model.
    full_results = run_hmm(df, ALL_FEATURES, "all_features")

    # Per feature models.
    feature_comparison = []
    for feature in ALL_FEATURES:
        res = run_hmm(df, [feature], f"single_{feature}")
        res["feature"] = feature
        feature_comparison.append(res)

    comp_df = pd.concat(feature_comparison)
    comp_df.to_csv(f"{BASE_OUTPUT}/feature_comparison.csv", index=False)

    print("\nDone.")

if __name__ == "__main__":
    main()