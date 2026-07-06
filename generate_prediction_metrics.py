#!/usr/bin/env python3
"""Generate metrics and error plots from a combined prediction CSV.

This is a script version of the metrics section in
``eqcctpro/notebooks/compile_results.ipynb``. It expects a CSV file such as ``eqcctpro/results/csv/combined/eqcct_explosions_th0_1_full.csv`` with prediction outputs combined via running ``combine_predictions.py``.
When a matching ground-truth CSV exists, it computes the same metrics and prediction-error plots used in the notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS_ROOT = Path("eqcctpro/results/metrics")
DEFAULT_GROUND_TRUTH_DIR = Path("eqcctpro/results/csv/ground_truth")
DEFAULT_ERROR_XLIM = (-0.5, 0.5)
DEFAULT_CDF_XLIM = (0.0, 0.5)


def infer_run_name(prediction_csv: Path) -> str:
    """Infer run name from a combined prediction filename."""

    stem = prediction_csv.stem
    if stem.endswith("_full"):
        stem = stem[: -len("_full")]
    return stem


def split_run_name(run_name: str) -> tuple[str, str]:
    """Return the filename prefix and threshold suffix used by notebook plots."""

    marker = "_th"
    if marker in run_name:
        prefix, suffix = run_name.split(marker, 1)
        return prefix, f"th{suffix}"
    return run_name, "metrics"


def normalize_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns in the combined prediction output."""

    df = df.copy()

    if "file_name" in df.columns:
        mseed_pattern = (
            r"(?P<station>[^/]+)/(?P<network>[^.]+)\.(?P=station)\.\.(?P<channel>[A-Z0-9]+)"
            r"__(?P<start_dt>\d{8}T\d{6}Z)__(?P<end_dt>\d{8}T\d{6}Z)\.mseed"
        )
        matches = df["file_name"].astype(str).str.extract(mseed_pattern)
        if matches.notna().all().all():
            df["network"] = matches["network"].str.strip().str.upper()
            df["station"] = matches["station"].str.strip().str.upper()

    for column in ("network", "station"):
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.upper()

    for column in ("p_arrival_time", "s_arrival_time", "start_dt", "end_dt"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if {"p_probability", "s_probability"}.issubset(df.columns):
        df = df[df["p_probability"].notna() | df["s_probability"].notna()]

    if "start_dt" in df.columns:
        df = df.sort_values(by="start_dt", ascending=True)

    return df


def prepare_long_predictions(df: pd.DataFrame, p_threshold: float) -> pd.DataFrame:
    """Convert wide prediction rows into one row per predicted P/S pick."""

    required = {
        "network",
        "station",
        "p_arrival_time",
        "p_probability",
        "s_arrival_time",
        "s_probability",
        "start_dt",
        "end_dt",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")

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
    df_pred_long["pred_arrival_time"] = pd.to_datetime(df_pred_long["pred_arrival_time"], errors="coerce")
    df_pred_long = df_pred_long.dropna(subset=["pred_arrival_time"])
    return df_pred_long.sort_values("pred_arrival_time")


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load and normalize a ground-truth pick CSV."""

    df_true = pd.read_csv(path)
    required = {"true_arrival_time", "network", "station", "phase"}
    missing = required.difference(df_true.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV is missing required columns: {sorted(missing)}")

    df_true = df_true.copy()
    df_true["true_arrival_time"] = pd.to_datetime(df_true["true_arrival_time"], errors="coerce")
    df_true["network"] = df_true["network"].astype(str).str.strip().str.upper()
    df_true["station"] = df_true["station"].astype(str).str.strip().str.upper()
    df_true["phase"] = df_true["phase"].astype(str).str.strip().str.upper()
    df_true = df_true[df_true["phase"].isin(["P", "S"])]
    df_true = df_true.dropna(subset=["true_arrival_time"])
    return df_true.sort_values("true_arrival_time")


def filter_truth_to_prediction_windows(
    df_true: pd.DataFrame,
    prediction_windows: pd.DataFrame,
    *,
    time_threshold: str = "0s",
) -> pd.DataFrame:
    """Keep true picks covered by prediction windows, plus match tolerance."""

    if df_true.empty or prediction_windows.empty:
        return df_true.iloc[0:0].copy()

    required_windows = {"network", "station", "start_dt", "end_dt"}
    missing_windows = required_windows.difference(prediction_windows.columns)
    if missing_windows:
        raise ValueError(f"Prediction CSV is missing window columns: {sorted(missing_windows)}")

    tol = pd.Timedelta(time_threshold)

    truth = df_true.copy()
    truth["true_arrival_time"] = pd.to_datetime(truth["true_arrival_time"], errors="coerce")
    truth["network"] = truth["network"].astype(str).str.strip().str.upper()
    truth["station"] = truth["station"].astype(str).str.strip().str.upper()
    truth = truth.dropna(subset=["true_arrival_time", "network", "station"])

    windows = prediction_windows[list(required_windows)].copy()
    windows["start_dt"] = pd.to_datetime(windows["start_dt"], errors="coerce")
    windows["end_dt"] = pd.to_datetime(windows["end_dt"], errors="coerce")
    windows["network"] = windows["network"].astype(str).str.strip().str.upper()
    windows["station"] = windows["station"].astype(str).str.strip().str.upper()
    windows = windows.dropna(subset=["start_dt", "end_dt", "network", "station"]).drop_duplicates()

    keep = pd.Series(False, index=truth.index)
    for row in windows.itertuples(index=False):
        keep |= (
            (truth["network"] == row.network)
            & (truth["station"] == row.station)
            & (truth["true_arrival_time"] >= row.start_dt - tol)
            & (truth["true_arrival_time"] <= row.end_dt + tol)
        )

    return truth[keep].drop_duplicates().sort_values("true_arrival_time")


def merge_predictions_and_truth(
    df_pred_long: pd.DataFrame,
    df_true: pd.DataFrame,
    time_threshold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match predictions to true picks using the notebook's merge-asof approach."""

    delta = pd.Timedelta(time_threshold)
    df_pred_long = df_pred_long.sort_values("pred_arrival_time")
    df_true = df_true.sort_values("true_arrival_time")

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

    picks_on_preds = picks_on_preds.sort_values(
        by=["network", "station", "phase", "true_arrival_time", "pred_arrival_time"]
    )
    picks_on_preds["time_diff"] = (
        picks_on_preds["pred_arrival_time"] - picks_on_preds["true_arrival_time"]
    ).abs()
    picks_on_preds = picks_on_preds.sort_values("time_diff").drop_duplicates(
        subset=["network", "station", "phase", "true_arrival_time"]
    )
    return preds_on_picks, picks_on_preds


def compute_metrics(preds_on_picks: pd.DataFrame, picks_on_preds: pd.DataFrame) -> dict[str, float]:
    """Compute overall and per-phase metrics in the same style as the notebook."""

    num_picks = len(picks_on_preds)
    num_preds = len(preds_on_picks)

    tp = int(picks_on_preds["probability"].notna().sum()) if "probability" in picks_on_preds else 0
    fp = int(num_preds - tp)
    fn = int(num_picks - tp)

    p_picks_on_preds = picks_on_preds[picks_on_preds["phase"] == "P"] if "phase" in picks_on_preds else pd.DataFrame()
    s_picks_on_preds = picks_on_preds[picks_on_preds["phase"] == "S"] if "phase" in picks_on_preds else pd.DataFrame()
    p_preds_on_picks = preds_on_picks[preds_on_picks["phase"] == "P"] if "phase" in preds_on_picks else pd.DataFrame()
    s_preds_on_picks = preds_on_picks[preds_on_picks["phase"] == "S"] if "phase" in preds_on_picks else pd.DataFrame()

    tp_p = int(p_picks_on_preds["probability"].notna().sum()) if "probability" in p_picks_on_preds else 0
    tp_s = int(s_picks_on_preds["probability"].notna().sum()) if "probability" in s_picks_on_preds else 0
    fp_p = int(len(p_preds_on_picks) - tp_p)
    fp_s = int(len(s_preds_on_picks) - tp_s)
    fn_p = int(len(p_picks_on_preds) - tp_p)
    fn_s = int(len(s_picks_on_preds) - tp_s)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "num_preds": num_preds,
        "num_picks": num_picks,
        "recall": divide(tp, tp + fn),
        "precision": divide(tp, tp + fp),
        "f1_score": f1_score(tp, fp, fn),
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
        "recall_p": divide(tp_p, tp_p + fn_p),
        "recall_s": divide(tp_s, tp_s + fn_s),
        "precision_p": divide(tp_p, tp_p + fp_p),
        "precision_s": divide(tp_s, tp_s + fp_s),
        "f1_score_p": f1_score(tp_p, fp_p, fn_p),
        "f1_score_s": f1_score(tp_s, fp_s, fn_s),
    }


def divide(numerator: float, denominator: float) -> float:
    """Divide with NaN for undefined values."""

    return numerator / denominator if denominator else np.nan


def f1_score(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from counts."""

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else np.nan


def format_metrics_report(metrics: dict[str, float], run_name: str, note: str | None = None) -> str:
    """Create the text report used by the notebook."""

    suffix = f"\nNote: {note}\n" if note else ""
    return f"""Evaluated predictions from: {run_name}{suffix}

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


def empty_match_frame(df_pred_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create metric inputs for runs with no ground truth, such as noise."""

    preds_on_picks = df_pred_long.copy()
    preds_on_picks["true_arrival_time"] = pd.NaT
    picks_on_preds = pd.DataFrame(
        columns=[
            "true_arrival_time",
            "network",
            "station",
            "phase",
            "channel",
            "pred_arrival_time",
            "probability",
            "start_dt",
            "end_dt",
            "time_diff",
        ]
    )
    return preds_on_picks, picks_on_preds


def save_error_plot(
    match_df: pd.DataFrame,
    phase: str,
    output_path: Path,
    time_threshold: str,
    p_threshold: float,
    error_xlim: tuple[float, float] | None = None,
    cdf_xlim: tuple[float, float] | None = None,
) -> None:
    """Save the histogram and cumulative-error plot for one phase."""

    label = "P-picks" if phase == "P" else "S-picks"
    df_phase = match_df[match_df["phase"] == phase].copy() if "phase" in match_df else pd.DataFrame()

    if not df_phase.empty and {"pred_arrival_time", "true_arrival_time"}.issubset(df_phase.columns):
        df_phase["error_seconds"] = (
            df_phase["pred_arrival_time"] - df_phase["true_arrival_time"]
        ).dt.total_seconds()
        errors = df_phase["error_seconds"].dropna()
    else:
        errors = pd.Series(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if errors.empty:
        text_string = f"No matched {label} errors\n\nDelta T = {time_threshold}\np_threshold = {p_threshold}"
        axes[0].text(
            0.5,
            0.5,
            text_string,
            transform=axes[0].transAxes,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(facecolor="w", alpha=0.8),
        )
        axes[0].set_title(f"Prediction Error on {label}")
        axes[0].set_xlabel("Error (s)")
        axes[0].set_ylabel("Count")
        if error_xlim is not None:
            axes[0].set_xlim(error_xlim)
        axes[1].set_title(f"{label} - cumulative error")
        axes[1].set_xlabel("Error (s)")
        if cdf_xlim is not None:
            axes[1].set_xlim(cdf_xlim)
        axes[1].grid()
    else:
        text_string = (
            f"$\\mu={errors.mean():.4f}$\n"
            f"$\\sigma={errors.std():.4f}$\n\n"
            f"Delta T = {time_threshold}\n"
            f"p_threshold = {p_threshold}"
        )

        axes[0].hist(errors, color="lightblue", edgecolor="black", bins=25)
        axes[0].axvline(errors.mean(), linestyle="dashed", linewidth=1, color="r", label="Mean")
        axes[0].axvline(0, linewidth=2, color="navy", label="Zero")
        axes[0].text(
            0.70,
            0.95,
            text_string,
            transform=axes[0].transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="w", alpha=0.8),
        )
        axes[0].set_title(f"Prediction Error on {label}")
        axes[0].set_xlabel("Error (s)")
        axes[0].set_ylabel("Count")
        if error_xlim is not None:
            axes[0].set_xlim(error_xlim)
        axes[0].legend(loc="lower right")

        sorted_err = np.sort(np.abs(errors))
        cdf = np.arange(len(sorted_err)) / len(sorted_err)
        axes[1].plot(sorted_err, cdf)
        axes[1].set_title(f"{label} - cumulative error")
        axes[1].set_xlabel("Error (s)")
        if cdf_xlim is not None:
            axes[1].set_xlim(cdf_xlim)
        axes[1].grid()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate notebook-style metrics and P/S error plots from a combined prediction CSV."
    )
    parser.add_argument("prediction_csv", type=Path, help="Combined prediction CSV to evaluate.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Ground-truth CSV. Defaults to eqcctpro/results/csv/ground_truth/<run_name>.csv when present.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help=f"Directory used for auto-detected ground truth. Default: {DEFAULT_GROUND_TRUTH_DIR}",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=DEFAULT_METRICS_ROOT,
        help=f"Root output directory for metrics. Default: {DEFAULT_METRICS_ROOT}",
    )
    parser.add_argument("--run-name", default=None, help="Override run name inferred from the CSV filename.")
    parser.add_argument("--p-threshold", type=float, default=0.1, help="Prediction probability threshold.")
    parser.add_argument("--time-threshold", default="0.5s", help="Maximum prediction/pick match window.")
    parser.add_argument(
        "--error-xlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_ERROR_XLIM,
        help=(
            "X-axis limits for the signed-error histogram. "
            f"Default: {DEFAULT_ERROR_XLIM[0]} {DEFAULT_ERROR_XLIM[1]}"
        ),
    )
    parser.add_argument(
        "--cdf-xlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_CDF_XLIM,
        help=(
            "X-axis limits for the absolute-error CDF. "
            f"Default: {DEFAULT_CDF_XLIM[0]} {DEFAULT_CDF_XLIM[1]}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prediction_csv = args.prediction_csv
    if not prediction_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv}")

    run_name = args.run_name or infer_run_name(prediction_csv)
    metrics_dir = args.metrics_root / run_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_prediction_frame(pd.read_csv(prediction_csv))
    df_pred_long = prepare_long_predictions(df, args.p_threshold)

    ground_truth_path = args.ground_truth
    if ground_truth_path is None:
        candidate = args.ground_truth_dir / f"{run_name}.csv"
        ground_truth_path = candidate if candidate.exists() else None

    note = None
    if ground_truth_path is None:
        preds_on_picks, picks_on_preds = empty_match_frame(df_pred_long)
        note = "No ground-truth CSV was found; all thresholded predictions are counted as false positives."
    else:
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground-truth CSV not found: {ground_truth_path}")
        df_true = load_ground_truth(ground_truth_path)
        df_true = filter_truth_to_prediction_windows(
            df_true,
            df,
            time_threshold=args.time_threshold,
        )
        preds_on_picks, picks_on_preds = merge_predictions_and_truth(
            df_pred_long,
            df_true,
            args.time_threshold,
        )

    metrics = compute_metrics(preds_on_picks, picks_on_preds)
    threshold_note = f"p_threshold={args.p_threshold}, time_threshold={args.time_threshold}"
    note = threshold_note if note is None else f"{threshold_note}. {note}"
    report = format_metrics_report(metrics, run_name, note=note)

    metrics_txt = metrics_dir / f"{run_name}.txt"
    metrics_txt.write_text(report)

    plot_prefix, plot_suffix = split_run_name(run_name)
    p_plot = metrics_dir / f"{plot_prefix}_p_picks_{plot_suffix}.png"
    s_plot = metrics_dir / f"{plot_prefix}_s_picks_{plot_suffix}.png"
    error_xlim = tuple(args.error_xlim) if args.error_xlim is not None else None
    cdf_xlim = tuple(args.cdf_xlim) if args.cdf_xlim is not None else None
    save_error_plot(picks_on_preds, "P", p_plot, args.time_threshold, args.p_threshold, error_xlim, cdf_xlim)
    save_error_plot(picks_on_preds, "S", s_plot, args.time_threshold, args.p_threshold, error_xlim, cdf_xlim)

    print(f"Wrote metrics report: {metrics_txt}")
    print(f"Wrote P-pick error plot: {p_plot}")
    print(f"Wrote S-pick error plot: {s_plot}")


if __name__ == "__main__":
    main()
