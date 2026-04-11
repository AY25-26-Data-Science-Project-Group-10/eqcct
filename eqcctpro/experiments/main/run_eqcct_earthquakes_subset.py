import os
import sys
import time
from datetime import datetime

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from eqcctpro import RunEQCCTPro

input_mseed_directory_path = os.path.join(base_dir, "data/waveforms_earthquakes_nonoise")
output_root_directory_path = os.path.join(base_dir, "results/csv/eqcct_earthquakes_subset")
models_dir = os.path.join(base_dir, "models/EQCCT")
tmp_dir = os.path.join(base_dir, "tmp")

# Timestamp-folder subset to run from the dataset.
# Uses Python slicing semantics: [SUBSET_START_INDEX:SUBSET_END_INDEX_EXCLUSIVE].
# Default: first 10 folders.
SUBSET_START_INDEX = 0
SUBSET_END_INDEX_EXCLUSIVE = 10

# Optional: run exactly one timestamp folder by name (set to None to disable).
# Example: "20250106T122712Z_20250106T122812Z"
SPECIFIC_TIMESTAMP_FOLDER = None

# Prediction threshold for both P and S picks.
PICK_THRESHOLD = 0.1

# CHANGE MARKER [EQCCT_TRACE_OUTPUT]:
# Optional EQCCT probability-traces.
SAVE_PROBABILITY_TRACES = True
PLOT_PROBABILITY_TRACES = True

os.makedirs(output_root_directory_path, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

cpu_workers = min(4, os.cpu_count() or 1)


def parse_chunk_dir_name(chunk_dir_name: str):
    """Parse 'YYYYMMDDTHHMMSSZ_YYYYMMDDTHHMMSSZ' into run-ready datetime strings."""
    start_raw, end_raw = chunk_dir_name.split("_")
    start_dt = datetime.strptime(start_raw, "%Y%m%dT%H%M%SZ")
    end_dt = datetime.strptime(end_raw, "%Y%m%dT%H%M%SZ")
    return start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S")


chunk_dirs = sorted(
    d for d in os.listdir(input_mseed_directory_path)
    if os.path.isdir(os.path.join(input_mseed_directory_path, d))
)
if SUBSET_START_INDEX < 0:
    raise ValueError("SUBSET_START_INDEX must be >= 0.")
if (
    SUBSET_END_INDEX_EXCLUSIVE is not None
    and SUBSET_END_INDEX_EXCLUSIVE < SUBSET_START_INDEX
):
    raise ValueError("SUBSET_END_INDEX_EXCLUSIVE must be >= SUBSET_START_INDEX.")

if SPECIFIC_TIMESTAMP_FOLDER is not None:
    if SPECIFIC_TIMESTAMP_FOLDER not in chunk_dirs:
        raise ValueError(
            f"SPECIFIC_TIMESTAMP_FOLDER '{SPECIFIC_TIMESTAMP_FOLDER}' was not found under "
            f"{input_mseed_directory_path}"
        )
    selected_chunk_dirs = [SPECIFIC_TIMESTAMP_FOLDER]
else:
    selected_chunk_dirs = chunk_dirs[SUBSET_START_INDEX:SUBSET_END_INDEX_EXCLUSIVE]

if not selected_chunk_dirs:
    raise RuntimeError(f"No timestamp folders found under: {input_mseed_directory_path}")

print(f"Running {len(selected_chunk_dirs)} timestamp folder(s) from: {input_mseed_directory_path}")
run_start = time.time()
for idx, chunk_dir_name in enumerate(selected_chunk_dirs, start=1):
    start_time, end_time = parse_chunk_dir_name(chunk_dir_name)
    out_dir = os.path.join(output_root_directory_path, chunk_dir_name)
    log_file_path = os.path.join(out_dir, "run.log")
    os.makedirs(out_dir, exist_ok=True)
    chunk_dir_path = os.path.join(input_mseed_directory_path, chunk_dir_name)
    chunk_stations = sorted(
        d for d in os.listdir(chunk_dir_path)
        if os.path.isdir(os.path.join(chunk_dir_path, d))
    )
    if not chunk_stations:
        print(f"[{idx}/{len(selected_chunk_dirs)}] {chunk_dir_name} has no station subdirs. Skipping.")
        continue
    specific_stations = ",".join(chunk_stations)

    print(
        f"[{idx}/{len(selected_chunk_dirs)}] {chunk_dir_name}  "
        f"({start_time} -> {end_time}), stations={len(chunk_stations)}"
    )

    runner_eqcct = RunEQCCTPro(
        use_gpu=False,
        model_type="eqcct",
        p_model_filepath=os.path.join(models_dir, "test_trainer_024.h5"),
        s_model_filepath=os.path.join(models_dir, "test_trainer_021.h5"),
        input_dir=input_mseed_directory_path,
        output_dir=out_dir,
        log_filepath=log_file_path,
        selected_gpus=[0],
        vram_mb=3000,
        cpu_id_list=range(0, cpu_workers),
        specific_stations=specific_stations,
        number_of_concurrent_station_predictions=1,
        number_of_concurrent_timechunk_predictions=1,
        P_threshold=PICK_THRESHOLD,
        S_threshold=PICK_THRESHOLD,
        save_probability_traces=SAVE_PROBABILITY_TRACES,
        plot_probability_traces=PLOT_PROBABILITY_TRACES,
        start_time=start_time,
        end_time=end_time,
        timechunk_dt=1,
        waveform_overlap=0,
        tmp_dir=tmp_dir,
    )
    runner_eqcct.run_eqcctpro()

total_seconds = time.time() - run_start
print(
    f"Finished predicting {len(selected_chunk_dirs)} timestamp folder(s) "
    f"in {total_seconds:.2f} s ({total_seconds/60.0:.2f} min)."
)
