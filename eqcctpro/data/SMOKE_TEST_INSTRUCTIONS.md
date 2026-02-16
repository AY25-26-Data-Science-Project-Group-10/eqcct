# EQCCTPro Smoke Test Instructions

This guide runs a smoke test with the 1-minute sample dataset.

## 1. Prepare Environment

From the repo root:

```bash
cd eqcctpro
conda activate eqcctpro-tf
```

## 2. Ensure Dataset Is Extracted

From `eqcctpro/data`:

```bash
unzip -o 230_stations_1_min_dt.zip
```

Expected extracted folder:

- `eqcctpro/data/230_stations_1_min_dt`

## 3. Run the Smoke Test

From `eqcctpro`:

```bash
python -m experiments.main.run_smoke_test
```

This script runs:

- `model_type='eqcct'`
- CPU only (`use_gpu=False`)
- 1-minute window (`2024-12-15 12:00:00` to `2024-12-15 12:01:00`)
- P model: `models/EQCCT/test_trainer_024.h5`
- S model: `models/EQCCT/test_trainer_021.h5`

## 4. Outputs

Smoke test output directory:

- `eqcctpro/results/csv/smoke_test/`

Log file:

- `eqcctpro/results/csv/smoke_test/eqcctpro_smoke_test.log`

