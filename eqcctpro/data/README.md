# Data Directory

This directory contains the seismic waveform data used by EQCCTPro.

## Structure
From the Official EQCCTPro Repository: 
- `230_stations_1_min_dt/`: A sample dataset containing 1-minute long mseed waveforms for 229 stations.
- `scripts/`: Contains `create_dataset.py`, which helps in downloading and organizing waveform data from FDSNWS sources into the format required by EQCCTPro.
- `archives/`: (Optional) Storage for compressed dataset files.

From DS Project:
- `waveforms_earthquakes_nonoise/` DS project dataset, consisting of 1-minute long mseed earthquake event waveforms from the Finnish seismic network from 01.01.2025 to 31.12.2025. Noise waveforms are not included. The corresponding labels are in `eq_labels.csv`
- `waveforms_explosions_nonoise/` DS project dataset, consisting of 1-minute long mseed explosion waveforms from the Finnish seismic network from 01.10.2025 to 28.02.2025. Noise waveforms are not included. The corresponding labels are in `ex_labels.csv`

## Input Format
EQCCTPro expects waveforms to be organized by time-chunk subdirectories, each containing station-specific subdirectories with three-component mseed files.
