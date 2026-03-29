import os
import sys
import time
import logging
from datetime import datetime

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from eqcctpro import RunEQCCTPro

input_mseed_directory_path = os.path.join(base_dir, "data/waveforms_noise_only")
output_root_directory_path = os.path.join(base_dir, "results/csv/eqcct_noise_th0_1")
models_dir = os.path.join(base_dir, "models/EQCCT")
tmp_dir = os.path.join(base_dir, "tmp")

# Default subset size for noise-only benchmark folders.
N_SUBSET_WAVEFORMS = 2000

os.makedirs(output_root_directory_path, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

cpu_workers = min(4, os.cpu_count() or 1)
baseline_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")


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
selected_chunk_dirs = chunk_dirs[:N_SUBSET_WAVEFORMS]

if not selected_chunk_dirs:
    raise RuntimeError(f"No timestamp folders found under: {input_mseed_directory_path}")

print(f"Running {len(selected_chunk_dirs)} timestamp folder(s) from: {input_mseed_directory_path}")
run_start = time.time()
failed_predictions = 0
skipped_folders = 0
failed_folder_names = []
for idx, chunk_dir_name in enumerate(selected_chunk_dirs, start=1):
    # Reset env that EQCCTPro mutates per run to prevent cumulative growth.
    os.environ["LD_LIBRARY_PATH"] = baseline_ld_library_path

    # Ensure each folder gets its own file handler.
    eqcctpro_logger = logging.getLogger("eqcctpro")
    for handler in list(eqcctpro_logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            eqcctpro_logger.removeHandler(handler)

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
        skipped_folders += 1
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
        vram_mb=4000,
        cpu_id_list=range(0, cpu_workers),
        specific_stations=specific_stations,
        number_of_concurrent_station_predictions=1,
        number_of_concurrent_timechunk_predictions=1,
        P_threshold=0.1,
        S_threshold=0.1,
        start_time=start_time,
        end_time=end_time,
        timechunk_dt=1,
        waveform_overlap=0,
        tmp_dir=tmp_dir,
    )
    try:
        runner_eqcct.run_eqcctpro()
    except Exception as exc:
        failed_predictions += 1
        failed_folder_names.append(chunk_dir_name)
        print(f"[{idx}/{len(selected_chunk_dirs)}] {chunk_dir_name} failed: {exc}")

total_seconds = time.time() - run_start
total_minutes = total_seconds / 60.0
successful_predictions = len(selected_chunk_dirs) - failed_predictions - skipped_folders
print(
    f"Finished predicting {len(selected_chunk_dirs)} timestamp folder(s) in {total_minutes:.2f} min. "
    f"Succeeded={successful_predictions}, Failed={failed_predictions}, Skipped={skipped_folders}."
)
if failed_folder_names:
    print("Failed folders:")
    for folder_name in failed_folder_names:
        print(f"- {folder_name}")
