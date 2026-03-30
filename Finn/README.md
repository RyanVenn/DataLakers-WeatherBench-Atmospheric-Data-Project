# Overview

My exploratory analysis started focussing on the [weatherbench subset](./weatherbench/), then [ghs_ucdb](./ghs_ucdb/), and a focus on [New York](./new_york/) via a combination of the latter + em-dat.

# Data Loading

Each of the aforementioned directories contains at least one `data_loader` script, which expect to be called from the root (in this case from the `Finn` directory) with the datasets stored in `Finn/datasets/...`.

## GHS_UCDB

First download the full dataset from https://human-settlement.emergency.copernicus.eu/download.php?ds=ucdb by clicking "Full Dataset".

The data can then be combined using the `data_extractor.py` script found [here](./ghs_ucdb/analysis/data_extractor.py) which expects the full dataset to be at `datasets/ghs_ucdb/GHS_UCDB_GLOBE_R2024A.xlsx`. It will save the merged excel sheets as CSVs at `datasets/ghs_ucdb/ucdb_merged.csv`, but only the columns I deemed relevant are extracted. Note that this will take a while to run.

The data loader is found [here](./ghs_ucdb/analysis/data_loader.py) and merges the output of `data_extractor.py` with the `gpkg` file (which stores spatial data). 

You can see the visualisation scripts for how to use the data loader.