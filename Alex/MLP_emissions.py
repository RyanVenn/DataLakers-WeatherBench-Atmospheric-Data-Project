import argparse
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="merged_all_locations_yearly_1990_2018.csv")
parser.add_argument("--target", required=True, choices=["co2_emissions_ton","ghg_emissions_ton","nox_emissions_ton","pm25_emissions_ton"])

args = parser.parse_args()

# Load data
df = pd.read_csv(args.csv)
df.columns = ["year","temp_mean_K","wind_speed_mean_ms","precip_total_mm","location_code","city","country","population","built_up_area_m2","gdp_ppp","hdi","pop_exposed_flood_10yr","co2_emissions_ton","ghg_emissions_ton","nox_emissions_ton","pm25_emissions_ton"]


# Train on 1990-2010 with all data visible
train = df[df["year"] <= 2010].copy()
test = df[df["year"] > 2010].copy()
training_columns = ["temp_mean_K", "wind_speed_mean_ms", "built_up_area_m2", "gdp_ppp", "population"]



# BEST FOR pm25: training_columns = ["wind_speed_mean_ms", "population", "built_up_area_m2", "pop_exposed_flood_10yr"]
# BEST FOR ghg: training_columns = ["wind_speed_mean_ms", "population", "built_up_area_m2",]
# BEST FOR co2: training_columns = ["temp_mean_K", "precip_total_mm", "built_up_area_m2", "gdp_ppp"]
# BEST FOR nox: training_columns = ["temp_mean_K", "wind_speed_mean_ms", "built_up_area_m2", "gdp_ppp", "population"]


# Our X is made up of the non-hidden variables and Y is the one we predict
X_train = train[training_columns].values
y_train = train[args.target].values

X_test = test[training_columns].values
y_test = test[args.target].values

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled = scaler.transform(X_test)         # use same scaler on test

model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    max_iter=20000,
    random_state=42
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {mae:.4f}")