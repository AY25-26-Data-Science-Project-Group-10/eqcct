import os
from eqcctpro import RunEQCCTPro

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

input_mseed_directory_path = os.path.join(base_dir, "data/230_stations_1_min_dt")
output_pick_directory_path = os.path.join(base_dir, "results/csv/smoke_test")
log_file_path = os.path.join(output_pick_directory_path, "eqcctpro_smoke_test.log")
models_dir = os.path.join(base_dir, "models/EQCCT")
tmp_dir = "/tmp"

os.makedirs(output_pick_directory_path, exist_ok=True)

cpu_workers = min(4, os.cpu_count() or 1)

runner_eqcct = RunEQCCTPro(
    use_gpu=False,
    model_type="eqcct",
    p_model_filepath=os.path.join(models_dir, "test_trainer_024.h5"),
    s_model_filepath=os.path.join(models_dir, "test_trainer_021.h5"),
    input_dir=input_mseed_directory_path,
    output_dir=output_pick_directory_path,
    log_filepath=log_file_path,
    cpu_id_list=range(0, cpu_workers),
    number_of_concurrent_station_predictions=1,
    number_of_concurrent_timechunk_predictions=1,
    P_threshold=0.001,
    S_threshold=0.02,
    start_time="2024-12-15 12:00:00",
    end_time="2024-12-15 12:01:00",
    timechunk_dt=1,
    waveform_overlap=0,
    tmp_dir=tmp_dir,
)
runner_eqcct.run_eqcctpro()
