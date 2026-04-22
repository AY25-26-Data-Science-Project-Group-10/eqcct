"""Reusable helpers for compile_results notebooks.

This module collects the parts of the notebook that are useful to import from
other notebooks or scripts:

- loading and normalizing prediction CSV files
- building waveform indexes and loading waveform windows
- plotting prediction rows and matched pick windows
- compiling ground-truth picks from QuakeML catalogs
- matching predictions to true picks and computing metrics
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read, read_events


P_PHASE_ALIASES = {"P", "PG", "PB", "PN"}
S_PHASE_ALIASES = {"S", "SG", "SB", "SN"}


def load_prediction_csvs(
    base_path: str,
    subdir: str,
    *,
    csv_output_path: str | None = None,
    csv_output_name: str | None = None,
) -> pd.DataFrame:
    """Read all prediction CSV files under a subdirectory and combine them.

    Parameters
    ----------
    base_path:
        Parent path containing the result folders.
    subdir:
        Subdirectory that contains the individual CSV files.
    csv_output_path, csv_output_name:
        Optional location to store the combined CSV.
    """

    files = list(Path(base_path, subdir).rglob("*.csv"))
    files = sorted(files, key=lambda x: x.parents[1].name)

    pattern = re.compile(r"(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)")
    dfs = []

    for file in files:
        df_tmp = pd.read_csv(file)
        folder = file.parents[1].name

        m = pattern.match(folder)
        if not m:
            raise ValueError(f"Could not parse folder name: {folder}")

        start_dt = pd.to_datetime(m.group("start"), format="%Y%m%dT%H%M%SZ")
        end_dt = pd.to_datetime(m.group("end"), format="%Y%m%dT%H%M%SZ")

        df_tmp["start_dt"] = start_dt
        df_tmp["end_dt"] = end_dt
        df_tmp["source_file"] = str(file.relative_to(base_path))
        dfs.append(df_tmp)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = normalize_prediction_frame(df)

    if csv_output_path and csv_output_name:
        os.makedirs(csv_output_path, exist_ok=True)
        df.to_csv(os.path.join(csv_output_path, csv_output_name), index=False)

    return df


def normalize_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction columns to a consistent format."""

    df = df.copy()

    if "file_name" in df.columns:
        mseed_pattern = (
            r"(?P<station>[^/]+)/(?P<network>[^.]+)\.(?P=station)\.\.(?P<channel>[A-Z]{3})"
            r"__(?P<start_dt>\d{8}T\d{6}Z)__(?P<end_dt>\d{8}T\d{6}Z)\.mseed"
        )
        matches = df["file_name"].str.extract(mseed_pattern)
        if matches.notna().all().all():
            df["network"] = matches["network"].str.strip().str.upper()
            df["station"] = matches["station"].str.strip().str.upper()
        elif "network" in df.columns and "station" in df.columns:
            df["network"] = df["network"].str.strip().str.upper()
            df["station"] = df["station"].str.strip().str.upper()
    elif "network" in df.columns and "station" in df.columns:
        df["network"] = df["network"].str.strip().str.upper()
        df["station"] = df["station"].str.strip().str.upper()

    for column in ("p_arrival_time", "s_arrival_time"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if {"p_probability", "s_probability"}.issubset(df.columns):
        df = df[df["p_probability"].notna() | df["s_probability"].notna()]

    if "start_dt" in df.columns:
        df = df.sort_values(by="start_dt", ascending=True)

    return df


def build_waveform_index(data_dir: str) -> dict[tuple[str, str, str, str], list[str]]:
    """Index waveform files by network, station, and time window."""

    index: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)

    file_pattern = re.compile(
        r"(?P<network>[^.]+)\.(?P<station>[^.]+)\.\.(?P<channel>[A-Z0-9]+)"
        r"__(?P<start>\d{8}T\d{6}Z)__(?P<end>\d{8}T\d{6}Z)\.mseed"
    )
    dir_pattern = re.compile(r"(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)")

    for entry in os.scandir(data_dir):
        if not entry.is_dir():
            continue

        dir_match = dir_pattern.fullmatch(entry.name)
        if not dir_match:
            continue

        dir_start = dir_match.group("start")
        dir_end = dir_match.group("end")

        for root, _, files in os.walk(entry.path):
            for filename in files:
                if not filename.endswith(".mseed"):
                    continue

                filename = filename.strip()
                match = file_pattern.fullmatch(filename)
                if not match:
                    print("No match:", filename)
                    continue

                network = match.group("network")
                station = match.group("station")
                key = (network, station, dir_start, dir_end)
                index[key].append(os.path.join(root, filename))

    return index


def load_station_window(
    index: dict[tuple[str, str, str, str], list[str]],
    network: str,
    station: str,
    start_dt,
    end_dt,
) -> Stream:
    """Load and trim all traces for a station/time window."""

    start_str = UTCDateTime(start_dt).strftime("%Y%m%dT%H%M%SZ")
    end_str = UTCDateTime(end_dt).strftime("%Y%m%dT%H%M%SZ")
    key = (network, station, start_str, end_str)

    st = Stream()
    if key not in index:
        print("Missing waveform:", key)
        return st

    for path in index[key]:
        st += read(path)

    if len(st) == 0:
        return st

    start = UTCDateTime(start_dt)
    end = UTCDateTime(end_dt)
    st.merge(method=1, fill_value="interpolate")
    st.trim(start, end)
    return st


def plot_pred_row(row, waveform_idx, p_tr: float = 0.01):
    """Plot a single prediction row if it exceeds the probability threshold."""

    if not (
        (pd.notna(row.p_probability) and row.p_probability > p_tr)
        or (pd.notna(row.s_probability) and row.s_probability > p_tr)
    ):
        return False

    st = load_station_window(
        waveform_idx,
        row.network,
        row.station,
        row.start_dt,
        row.end_dt,
    )

    if len(st) == 0:
        print("No waveform:", row.station, row.start_dt)
        return False

    for tr in st:
        if tr.stats.sampling_rate > 100:
            tr.resample(100)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{row.network}.{row.station}  {row.start_dt} to {row.end_dt}", fontsize=10)

    channels = [tr.stats.channel for tr in st]

    for j, ch in enumerate(channels):
        tr_sel = st.select(channel=ch)
        if len(tr_sel) == 0:
            continue

        tr = tr_sel[0].copy()
        tr.detrend("linear")
        tr.detrend("demean")
        tr.filter("bandpass", freqmin=1, freqmax=45, corners=2, zerophase=True)
        tr.taper(max_percentage=0.01, type="cosine", max_length=2)
        tr.normalize()

        t = tr.times()
        axes[j].plot(t, tr.data, color="grey", linewidth=1)

        if pd.notna(row.p_arrival_time):
            time_pred = UTCDateTime(row.p_arrival_time) - tr.stats.starttime
            axes[j].axvline(
                time_pred,
                color="r",
                linewidth=1.5,
                linestyle="dashed",
                label=f"P-pred (p={row.p_probability:.3f})",
            )

        if pd.notna(row.s_arrival_time):
            time_pred = UTCDateTime(row.s_arrival_time) - tr.stats.starttime
            axes[j].axvline(
                time_pred,
                color="b",
                linewidth=1.5,
                linestyle="dashed",
                label=f"S-pred (p={row.s_probability:.3f})",
            )

        axes[j].set_ylabel(ch)
        axes[j].legend(loc="lower right")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    return True


def plot_window(group_df, waveform_idx, prob_tr: float = 0.01, delta_tr: float = 0, mode: str = "both"):
    """Plot waveform windows with predicted and true picks overlaid."""

    row0 = group_df.iloc[0]

    st = load_station_window(
        waveform_idx,
        row0.network,
        row0.station,
        row0.start_dt,
        row0.end_dt,
    )

    if len(st) == 0:
        print("No waveform")
        return False

    for tr in st:
        if tr.stats.sampling_rate > 100:
            tr.resample(100)

    channel_comp = ["Z"]
    plot_channels = []
    allowed_phases = {"both": {"P", "S"}, "p_picks": {"P"}, "s_picks": {"S"}}[mode]

    for ch in channel_comp:
        tr_sel = st.select(channel=f"*{ch}")
        if len(tr_sel) == 0:
            continue

        tr = tr_sel[0].copy()
        valid_picks = []
        p_pick = None
        s_pick = None

        for _, pick in group_df.iterrows():
            if pick.probability < prob_tr or pick.phase not in allowed_phases:
                continue

            time_pred = UTCDateTime(pick.pred_arrival_time) - tr.stats.starttime if pd.notna(pick.pred_arrival_time) else None
            time_true = UTCDateTime(pick.true_arrival_time) - tr.stats.starttime if pd.notna(pick.true_arrival_time) else None
            time_delta = time_true - time_pred if time_pred is not None and time_true is not None else None
            pred_prob = pick.probability

            if pick.phase == "P":
                p_pick = (pick, time_pred, time_true, time_delta, pred_prob)
            elif pick.phase == "S":
                s_pick = (pick, time_pred, time_true, time_delta, pred_prob)

        if delta_tr == 0:
            if p_pick is not None and p_pick[1] is not None:
                valid_picks.append(p_pick)
            if s_pick is not None and s_pick[1] is not None:
                valid_picks.append(s_pick)
        else:
            include_p = p_pick is not None and p_pick[3] is not None and abs(p_pick[3]) > delta_tr
            include_s = s_pick is not None and s_pick[3] is not None and abs(s_pick[3]) > delta_tr

            if include_p or include_s:
                if p_pick is not None:
                    valid_picks.append(p_pick)
                if s_pick is not None:
                    valid_picks.append(s_pick)

        if valid_picks:
            plot_channels.append((ch, tr_sel[0], valid_picks))

    if not plot_channels:
        return False

    fig, axes = plt.subplots(len(plot_channels), 1, figsize=(14, 3 * len(plot_channels)), sharex=True)
    if len(plot_channels) == 1:
        axes = [axes]

    fig.suptitle(f"{row0.network}.{row0.station}  {row0.start_dt} to {row0.end_dt}", fontsize=10)

    for j, (ch, tr, valid_picks) in enumerate(plot_channels):
        start_time = tr.stats.starttime

        tr = tr.copy()
        tr.detrend("linear")
        tr.detrend("demean")
        tr.filter("bandpass", freqmin=1, freqmax=45, corners=2, zerophase=True)
        tr.taper(max_percentage=0.01, type="cosine", max_length=2)
        tr.normalize()
        tr.trim(starttime=start_time + 3)

        axes[j].plot(tr.times(), tr.data, color="grey", linewidth=1)
        axes[j].set_ylabel(ch)

        for pick, time_pred, time_true, time_delta, prob in valid_picks:
            if pick.phase == "P":
                c = "r"
                text_x_pos = time_pred - 7 if time_pred is not None else 0
                text_y_pos = tr.data.max() * 0.7
            else:
                c = "b"
                text_x_pos = time_pred + 1 if time_pred is not None else 0
                text_y_pos = tr.data.min() * 0.8

            if time_pred is not None:
                axes[j].axvline(time_pred, color=c, linewidth=1.5, linestyle="dotted", label=f"{pick.phase}-predicted (p={prob})")
                print(f"Pred time: {start_time + time_pred}")

            if time_true is not None:
                axes[j].axvline(time_true, color=c, linewidth=1.5)
                print(f"Pick time: {start_time + time_true}")

            if time_delta is not None:
                axes[j].text(
                    x=text_x_pos,
                    y=text_y_pos,
                    s=f"delta: {time_delta:.2f}s",
                    bbox=dict(facecolor="w", alpha=1, edgecolor=c),
                )

        axes[j].legend(loc="lower right")
        axes[j].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()
    return True


def load_qml_catalogs(qml_dir: str | Path, filenames: dict[str, str] | None = None) -> dict[str, object]:
    """Load QuakeML catalogs from disk."""

    qml_dir = Path(qml_dir)
    if filenames is None:
        filenames = {
            "earthquakes": "earthquakes2025.qkml",
            "explosions": "explosions2025.qkml",
        }

    catalogs = {}
    for name, filename in filenames.items():
        path = qml_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.resolve()}")
        catalogs[name] = read_events(str(path))

    return catalogs


def compile_true_picks_from_catalog(
    eq_cat,
    prediction_df: pd.DataFrame | None = None,
    station_filter: dict[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    """Extract true P/S picks from a QuakeML catalog into a dataframe."""

    if station_filter is None and prediction_df is not None:
        station_filter = {
            ntwk: prediction_df[prediction_df["network"] == ntwk]["station"].unique().tolist()
            for ntwk in prediction_df["network"].unique()
        }

    event_ids = []
    times = []
    networks = []
    stations = []
    phases = []
    channels = []

    for event in eq_cat:
        event_id = str(event.resource_id)

        for pick in event.picks:
            if pick.waveform_id is None:
                continue

            channel = pick.waveform_id.channel_code
            pick_network = pick.waveform_id.network_code
            pick_station = pick.waveform_id.station_code
            pick_phase_hint = pick.phase_hint

            if station_filter is not None:
                if pick_network not in station_filter.keys() or pick_station not in station_filter[pick_network]:
                    continue

            if not pick_phase_hint:
                continue

            ph = (pick_phase_hint or "").upper().strip()
            if ph in P_PHASE_ALIASES:
                pick_phase = "P"
            elif ph in S_PHASE_ALIASES:
                pick_phase = "S"
            else:
                continue

            event_ids.append(event_id)
            times.append(pick.time.datetime)
            networks.append(pick_network)
            stations.append(pick_station)
            phases.append(pick_phase)
            channels.append(channel)

    df_true = pd.DataFrame(
        {
            "event_id": event_ids,
            "true_arrival_time": times,
            "network": networks,
            "station": stations,
            "phase": phases,
            "channel": channels,
        }
    )

    if not df_true.empty:
        df_true = df_true.sort_values(by="true_arrival_time", ascending=True)

    return df_true


def save_true_picks(df_true: pd.DataFrame, gt_path: str | Path, gt_fname: str) -> str:
    """Save a true-pick dataframe to disk and return the file path."""

    gt_path = Path(gt_path)
    gt_path.mkdir(parents=True, exist_ok=True)
    output_path = gt_path / gt_fname
    df_true.to_csv(output_path, index=False)
    return str(output_path)


def prepare_long_predictions(df: pd.DataFrame, p_threshold: float = 0.1) -> pd.DataFrame:
    """Convert wide prediction output to a long dataframe with one row per pick."""

    df = df.sort_values("start_dt")

    df_pred_p = df[["network", "station", "p_arrival_time", "p_probability", "start_dt", "end_dt"]].rename(
        columns={"p_arrival_time": "pred_arrival_time", "p_probability": "probability"}
    )
    df_pred_s = df[["network", "station", "s_arrival_time", "s_probability", "start_dt", "end_dt"]].rename(
        columns={"s_arrival_time": "pred_arrival_time", "s_probability": "probability"}
    )

    df_pred_p["phase"] = "P"
    df_pred_s["phase"] = "S"

    df_pred_long = pd.concat([df_pred_p, df_pred_s], ignore_index=True)
    df_pred_long = df_pred_long.dropna(subset=["pred_arrival_time", "probability"])
    df_pred_long = df_pred_long[df_pred_long["probability"] > p_threshold]
    df_pred_long["pred_arrival_time"] = pd.to_datetime(df_pred_long["pred_arrival_time"])
    return df_pred_long.sort_values("pred_arrival_time")


def merge_predictions_and_truth(
    df_pred_long: pd.DataFrame,
    df_true: pd.DataFrame,
    time_threshold: str = "10s",
):
    """Match predicted picks to true picks and return both merge directions."""

    df_true = df_true.copy()
    df_pred_long = df_pred_long.copy()

    df_true["true_arrival_time"] = pd.to_datetime(df_true["true_arrival_time"])
    df_true = df_true[df_true["phase"].isin(["P", "S"])]

    df_pred_long = df_pred_long.sort_values("pred_arrival_time")
    df_true = df_true.sort_values("true_arrival_time")

    delta = pd.Timedelta(time_threshold)

    preds_on_picks = pd.merge_asof(
        df_pred_long,
        df_true,
        left_on="pred_arrival_time",
        right_on="true_arrival_time",
        by=["network", "station", "phase"],
        tolerance=delta,
        direction="nearest",
    )

    picks_on_preds = pd.merge_asof(
        df_true,
        df_pred_long,
        left_on="true_arrival_time",
        right_on="pred_arrival_time",
        by=["network", "station", "phase"],
        tolerance=delta,
        direction="nearest",
    )

    picks_on_preds = picks_on_preds.sort_values(by=["network", "station", "phase", "true_arrival_time", "pred_arrival_time"])
    picks_on_preds["time_diff"] = (picks_on_preds["pred_arrival_time"] - picks_on_preds["true_arrival_time"]).abs()
    picks_on_preds = picks_on_preds.sort_values("time_diff").drop_duplicates(subset=["network", "station", "phase", "true_arrival_time"])

    return preds_on_picks, picks_on_preds


def compute_metrics(preds_on_picks: pd.DataFrame, picks_on_preds: pd.DataFrame) -> dict[str, float]:
    """Compute overall and per-phase metrics from matched dataframes."""

    num_picks = len(picks_on_preds)
    num_preds = len(preds_on_picks)

    tp = int(picks_on_preds["probability"].notna().sum())
    fp = int(num_preds - tp)
    fn = int(num_picks - tp)

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else np.nan

    p_picks_on_preds = picks_on_preds[picks_on_preds["phase"] == "P"]
    s_picks_on_preds = picks_on_preds[picks_on_preds["phase"] == "S"]
    p_preds_on_picks = preds_on_picks[preds_on_picks["phase"] == "P"]
    s_preds_on_picks = preds_on_picks[preds_on_picks["phase"] == "S"]

    tp_p = int(p_picks_on_preds["probability"].notna().sum())
    tp_s = int(s_picks_on_preds["probability"].notna().sum())
    fp_p = int(len(p_preds_on_picks) - tp_p)
    fp_s = int(len(s_preds_on_picks) - tp_s)
    fn_p = int(len(p_picks_on_preds) - tp_p)
    fn_s = int(len(s_picks_on_preds) - tp_s)

    recall_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) else np.nan
    recall_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else np.nan
    precision_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) else np.nan
    precision_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else np.nan
    f1_score_p = 2 * (precision_p * recall_p) / (precision_p + recall_p) if (precision_p + recall_p) else np.nan
    f1_score_s = 2 * (precision_s * recall_s) / (precision_s + recall_s) if (precision_s + recall_s) else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "num_preds": num_preds,
        "num_picks": num_picks,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
        "TP_p": tp_p,
        "TP_s": tp_s,
        "FP_p": fp_p,
        "FP_s": fp_s,
        "FN_p": fn_p,
        "FN_s": fn_s,
        "num_preds_p": len(p_preds_on_picks),
        "num_preds_s": len(s_preds_on_picks),
        "num_picks_p": len(p_picks_on_preds),
        "num_picks_s": len(s_picks_on_preds),
        "recall_p": recall_p,
        "recall_s": recall_s,
        "precision_p": precision_p,
        "precision_s": precision_s,
        "f1_score_p": f1_score_p,
        "f1_score_s": f1_score_s,
    }


def format_metrics_report(metrics: dict[str, float], subdir: str) -> str:
    """Create the text report used by the notebook."""

    return f"""Evaluated predictions from: {subdir}

=========== Overall metrics ===========
True Positives:        {metrics['TP']}
False Positives:       {metrics['FP']}
False Negatives:       {metrics['FN']}
Total predictions:     {metrics['num_preds']}
Total picks:           {metrics['num_picks']}
Recall:                {metrics['recall']}
Precision:             {metrics['precision']}
F1 Score:              {metrics['f1_score']}

=========== Metrics for p-picks ===========
True Positives:        {metrics['TP_p']}
False Positives:       {metrics['FP_p']}
False Negatives:       {metrics['FN_p']}
Total predictions:     {metrics['num_preds_p']}
Total picks:           {metrics['num_picks_p']}
Recall:                {metrics['recall_p']}
Precision:             {metrics['precision_p']}
F1 Score:              {metrics['f1_score_p']}

=========== Metrics for s-picks ===========
True Positives:        {metrics['TP_s']}
False Positives:       {metrics['FP_s']}
False Negatives:       {metrics['FN_s']}
Total predictions:     {metrics['num_preds_s']}
Total picks:           {metrics['num_picks_s']}
Recall:                {metrics['recall_s']}
Precision:             {metrics['precision_s']}
F1 Score:              {metrics['f1_score_s']}
"""


def save_metrics_report(metrics_string: str, metrics_dir: str | Path, filename: str) -> str:
    """Write a metrics report to disk and return the file path."""

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / filename
    output_path.write_text(metrics_string)
    return str(output_path)


def build_error_dataframe(match_df: pd.DataFrame) -> pd.DataFrame:
    """Add an error column in seconds to a matched dataframe."""

    df = match_df.copy()
    df["error_seconds"] = (df["pred_arrival_time"] - df["true_arrival_time"]).dt.total_seconds()
    return df
