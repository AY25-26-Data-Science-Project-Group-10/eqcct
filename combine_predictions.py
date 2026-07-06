import os
from pathlib import Path
import glob
import re
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
from obspy import read
import numpy as np
import matplotlib.pyplot as plt

# Define the model e.g. "eqcct", "eqt"
model_type = "eqcct"
# Define the type of waveforms used e.g. "earthquakes", "explosions", "noise"
waveform_type = "earthquakes"

# Path for .mseed waveforms
if waveform_type == "noise":
    data_dir = f"eqcctpro/data/waveforms_noise_only/"
else:
    data_dir = f"eqcctpro/data/waveforms_{waveform_type}_nonoise/"

# base path: where to gather the outputs from
base_path = "eqcctpro/results/csv/"

# subdir: subdirectory for the prediction results, based on model type and waveform type
subdir = f"{model_type}_{waveform_type}_th0_1"

# csv output path: where to store the combined csv
csv_output_path = os.path.join(base_path, "combined")
csv_output_name = subdir + "_full.csv"

# Match filenames to only select .csv files (ignore logs, etc.)
files = list(Path(base_path + subdir).rglob("*.csv"))

# Sort files by timechunk folder name (0 = station, 1 = timechunk)
files = sorted(files, key=lambda x:x.parents[1].name)

# Load files
dfs = []

# Start and end times of the time windows as in the folder names
pattern = re.compile(r"(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)")

for file in files:
    df_tmp = pd.read_csv(file)

    # Extract time window from folder name
    # e.g. .../20250101T092157Z_20250101T092257Z/HEF_outputs/X_prediction_results.csv
    folder = file.parents[1].name

    m = pattern.match(folder)
    if not m:
        raise ValueError(f"Could not parse folder name: {folder}")

    # Attach to dataframe
    df_tmp["start_dt"] = pd.to_datetime(m.group("start"), format="%Y%m%dT%H%M%SZ")
    df_tmp["end_dt"] = pd.to_datetime(m.group("end"), format="%Y%m%dT%H%M%SZ")
    df_tmp["source_file"] = str(file.relative_to(base_path))

    dfs.append(df_tmp)

if not dfs:
    raise ValueError(f"No .csv files found in: {base_path + subdir}")

# Combine into single df
df = pd.concat(dfs, ignore_index=True)

# Keep only rows with predictions
pred_cols = ["p_probability", "s_probability"]

# Extract raw datetime string from time window directories
mseed_pattern = r"(?P<station>[^/]+)/(?P<network>[^.]+)\.(?P=station)\.\.(?P<channel>[A-Z]{3})__(?P<start_dt>\d{8}T\d{6}Z)__(?P<end_dt>\d{8}T\d{6}Z)\.mseed"
matches = df["file_name"].str.extract(mseed_pattern)

# If filename information is present in predictions, use it to infer network and station codes
# This is done because eqcct assigns 0 as the network code for some reason
if matches.notna().all().all():
    df["network"] = matches["network"].str.strip().str.upper()
    df["station"] = matches["station"].str.strip().str.upper()

# With EQT, the .mseed filename is not available, but network and station should be correct in the predictions .csv
else:
    df["network"] = df["network"].str.strip().str.upper()
    df["station"] = df["station"].str.strip().str.upper()

# Convert arrivals to datetime
df["p_arrival_time"] = pd.to_datetime(df["p_arrival_time"], errors="coerce")
df["s_arrival_time"] = pd.to_datetime(df["s_arrival_time"], errors="coerce")

# Count the number of rows with NaN for both p_probability and s_probability (no predicted picks)
num_negatives = sum(df["p_probability"].isna() & df["s_probability"].isna())

# Filter out rows with NaN for both p_probability and s_probability (no predicted picks)
df = df[df["p_probability"].notna() | df["s_probability"].notna()]

# Sort df by the window start time
df = df.sort_values(by="start_dt", ascending=True)
# Save dataframe to csv
os.makedirs(csv_output_path, exist_ok=True)
df.to_csv(os.path.join(csv_output_path, csv_output_name), index=False)

# Print some basic information about the number of predicted picks
num_rows = len(df)
num_files = len(files)
num_p_picks = sum(df["p_probability"].notna())
num_s_picks = sum(df["s_probability"].notna())

print(f"Read {num_rows} rows from {num_files} files.")
print(f"Predictions stored in {os.path.join(csv_output_path, csv_output_name)}.")
print(f"Number of predicted p-arrivals: {num_p_picks}\nNumber of predicted s-arrivals: {num_s_picks}\nTotal number of arrivals: {num_p_picks+num_s_picks}")
print(f"Number of empty predictions: {num_negatives}")