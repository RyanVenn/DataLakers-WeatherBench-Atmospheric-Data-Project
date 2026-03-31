# Overview

My exploratory analysis started focussing on the [weatherbench subset](./weatherbench/), then [ghs_ucdb](./ghs_ucdb/), and a focus on [New York](./new_york/) via a combination of the latter + em-dat.

# Data Loading

Each of the aforementioned directories contains at least one `data_loader` script, which expect to be called from the root (in this case from the `Finn` directory) with the datasets stored in `Finn/datasets/...`.

## GHS_UCDB

First download the full dataset from https://human-settlement.emergency.copernicus.eu/download.php?ds=ucdb by clicking "Full Dataset".

The data can then be combined using the `data_extractor.py` script found [here](./ghs_ucdb/analysis/data_extractor.py) which expects the full dataset to be at `datasets/ghs_ucdb/GHS_UCDB_GLOBE_R2024A.xlsx`. It will save the merged excel sheets as CSVs at `datasets/ghs_ucdb/ucdb_merged.csv`, but only the columns I deemed relevant are extracted. Note that this will take a while to run.

The data loader is found [here](./ghs_ucdb/analysis/data_loader.py) and merges the output of `data_extractor.py` with the `gpkg` file (which stores spatial data). 

You can see the visualisation scripts for how to use the data loader.


# Dataset Building

The final dataset is built from our Weatherbench subset and UCDB. The following columns were merged (and renamed):
- `population`
- `built_up_area_m2`
- `gdp_ppp`
- `hdi`: Human Development Index.
- `pop_exposed_flood_10yr`: The number of people exposed to a flood with a 10-year return period (i.e., a flood that statistically occurs once every 10 years). This is an indicator of flood risk.
- `co2_emissions_ton`: Total CO₂ emissions (tons/year).
- `ghg_emissions_ton`: Total greenhouse gas emissions (tons CO₂‑equivalent/year).
- `nox_emissions_ton`: Total nitrogen oxides emissions (tons/year).
- `pm25_emissions_ton`: Total PM2.5 (fine particulate matter) emissions (tons/year).
- `temp_mean_K`
- `precip_total_mm`
- `wind_speed_mean_ms`: The wind speed is a derived feature where $s = \sqrt{s_u^2 + s_v^2}$.

The data from Weatherbench was hourly, whilst the data from UCDB was either annual or quinquennial (occurring every 5 years). Hence, we decided on preprocessing the data such that every column was annual:
- `temp_mean_K` and `wind_speed_mean_ms` were hourly and thus the mean was taken.
- `precip_total_mm` was hourly, however the total was used instead of the mean because it provides the absolute, cumulative amount of water deposited over a specific area and time, whereas the mean can be highly misleading due to the high variability and skewed nature of rainfall distribution.
- `popuplation`, `built_up_area_m2`, `gdp_ppp`, `hdi`, `pop_exposed_flood_10yr`, and all of the emmissions were quinquennial; thus the 4 years in between each record were interpolated.
- The remainder of the columns were already annual.

Note that `gdp_ppp` and `hdi` began in 1990 whereas the other columns started in either 1975 or 1980. 15 years is unreasonable for interpolation, so we decided to drop all records before 1990. 
