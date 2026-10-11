"""Evaluate leakage-safe hourly calibration for Experiment 5 no_graph predictions.

The script uses cached Experiment 5 no_graph detailed predictions and applies
expanding-window ratio calibration. For each target hour, calibration factors are
estimated only from earlier target hours, then applied to the current hour.

Outputs include raw-vs-calibrated metrics and detailed CSVs that can be passed
to the from_ex5 experiment scripts via --ex5-detail.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    BASE_DIR
    / "ex5_weight_csv"
    / "results"
    / "experiment_5_graph_structure_comparison_detailed.csv"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "peak_tcn_calibration"
DEFAULT_PEAK_HOURS = [6, 7, 8, 9]
StatsKey = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze hourly calibration for Experiment 5 no_graph predictions."
    )
    parser.add_argument("--detailed", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prediction-column",
        choices=["refined_pred", "base_pred"],
        default="refined_pred",
        help="Raw no_graph prediction column to calibrate.",
    )
    parser.add_argument("--peak-hours", type=int, nargs="*", default=DEFAULT_PEAK_HOURS)
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum historical rows required before using a group-specific factor.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Additive smoothing for ratio true_sum / pred_sum.",
    )
    parser.add_argument("--min-factor", type=float, default=0.2)
    parser.add_argument("--max-factor", type=float, default=5.0)
    parser.add_argument(
        "--eval-split",
        choices=["test", "all"],
        default="test",
        help="Rows used for summary metrics. Calibration fitting always uses earlier hours.",
    )
    args = parser.parse_args()
    if not args.detailed.exists():
        parser.error(f"--detailed does not exist: {args.detailed}")
    if args.min_count < 1:
        parser.error("--min-count must be >= 1.")
    if args.smoothing < 0:
        parser.error("--smoothing must be >= 0.")
    if args.min_factor <= 0 or args.max_factor <= 0 or args.min_factor > args.max_factor:
        parser.error("Require 0 < --min-factor <= --max-factor.")
    invalid_hours = [hour for hour in args.peak_hours if hour < 0 or hour > 23]
    if invalid_hours:
        parser.error(f"Invalid peak hour(s): {invalid_hours}")
    return args


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_no_graph_predictions(path: Path, prediction_column: str) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"target_hour", "graph_type", "PULocationID", "true_value", prediction_column}
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    preferred = [
        "target_hour",
        "graph_type",
        "mode",
        "experiment_3_mode",
        "Model",
        "random_seed",
        "edge_weight_used",
        "PULocationID",
        "split",
        "is_test",
        "base_pred",
        "true_value",
        "residual_target",
        "residual_pred",
        "refined_pred",
        "sample_weight",
        "prior_score",
        "stability_score",
        "history_consistency_score",
        "full_confidence",
        "best_epoch",
        "best_val_loss",
        "window_id",
        "window_start",
        "window_step",
        "w_prior",
        "w_stability",
        "w_history_consistency",
    ]
    usecols = [column for column in preferred if column in columns]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["target_hour"])
    df = df[df["graph_type"].astype(str).eq("no_graph")].copy()
    if df.empty:
        raise ValueError("No graph_type == 'no_graph' rows found.")
    df["PULocationID"] = df["PULocationID"].astype(int)
    df["raw_prediction"] = pd.to_numeric(df[prediction_column], errors="coerce")
    df["true_value"] = pd.to_numeric(df["true_value"], errors="coerce")
    df = df[
        np.isfinite(df["raw_prediction"].to_numpy(dtype=float))
        & np.isfinite(df["true_value"].to_numpy(dtype=float))
    ].copy()
    if df.empty:
        raise ValueError("No finite no_graph predictions found.")
    return df.sort_values(["target_hour", "PULocationID"]).reset_index(drop=True)


def empty_stat() -> list[float]:
    return [0.0, 0.0, 0.0]


def update_stat(stat: list[float], y_true: float, y_pred: float) -> None:
    stat[0] += float(y_true)
    stat[1] += float(y_pred)
    stat[2] += 1.0


def ratio_from_stat(
    stat: list[float],
    min_count: int,
    smoothing: float,
    min_factor: float,
    max_factor: float,
) -> float | None:
    true_sum, pred_sum, count = stat
    if count < min_count:
        return None
    denominator = pred_sum + smoothing
    if denominator <= 1e-9:
        return None
    factor = (true_sum + smoothing) / denominator
    return float(np.clip(factor, min_factor, max_factor))


def apply_expanding_calibration(
    df: pd.DataFrame,
    min_count: int,
    smoothing: float,
    min_factor: float,
    max_factor: float,
    peak_hours: set[int],
) -> pd.DataFrame:
    hour_stats: Dict[int, list[float]] = defaultdict(empty_stat)
    zone_hour_stats: Dict[StatsKey, list[float]] = defaultdict(empty_stat)
    global_stat = empty_stat()
    calibrated_frames: list[pd.DataFrame] = []

    for target_hour, hour_df in df.groupby("target_hour", sort=True):
        current = hour_df.copy()
        hour_value = int(pd.Timestamp(target_hour).hour)
        global_factor = ratio_from_stat(
            global_stat,
            min_count=min_count,
            smoothing=smoothing,
            min_factor=min_factor,
            max_factor=max_factor,
        )
        if global_factor is None:
            global_factor = 1.0

        hour_factor = ratio_from_stat(
            hour_stats[hour_value],
            min_count=min_count,
            smoothing=smoothing,
            min_factor=min_factor,
            max_factor=max_factor,
        )
        if hour_factor is None:
            hour_factor = global_factor

        zone_hour_factors = []
        for row in current.itertuples():
            key = (int(row.PULocationID), hour_value)
            factor = ratio_from_stat(
                zone_hour_stats[key],
                min_count=min_count,
                smoothing=smoothing,
                min_factor=min_factor,
                max_factor=max_factor,
            )
            zone_hour_factors.append(hour_factor if factor is None else factor)

        raw = current["raw_prediction"].to_numpy(dtype=float)
        current["global_factor"] = global_factor
        current["hour_factor"] = hour_factor
        current["zone_hour_factor"] = zone_hour_factors
        current["raw_no_graph"] = raw
        current["global_calibrated"] = np.maximum(0.0, raw * global_factor)
        current["hour_calibrated"] = np.maximum(0.0, raw * hour_factor)
        current["zone_hour_calibrated"] = np.maximum(
            0.0,
            raw * np.asarray(zone_hour_factors, dtype=float),
        )
        current["peak_hour_calibrated"] = np.where(
            hour_value in peak_hours,
            current["hour_calibrated"],
            current["raw_no_graph"],
        )
        calibrated_frames.append(current)

        for row in hour_df.itertuples():
            y_true = float(row.true_value)
            y_pred = float(row.raw_prediction)
            update_stat(global_stat, y_true=y_true, y_pred=y_pred)
            update_stat(hour_stats[hour_value], y_true=y_true, y_pred=y_pred)
            update_stat(
                zone_hour_stats[(int(row.PULocationID), hour_value)],
                y_true=y_true,
                y_pred=y_pred,
            )

    return pd.concat(calibrated_frames, ignore_index=True)


def metric_row(
    df: pd.DataFrame,
    variant: str,
    slice_name: str,
    pred_column: str,
) -> dict[str, object]:
    y_true = df["true_value"].to_numpy(dtype=float)
    y_pred = df[pred_column].to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {
            "variant": variant,
            "slice": slice_name,
            "n_predictions": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MSE": np.nan,
            "Bias": np.nan,
        }
    errors = y_pred[mask] - y_true[mask]
    mse = float(np.mean(errors**2))
    return {
        "variant": variant,
        "slice": slice_name,
        "n_predictions": int(mask.sum()),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
        "Bias": float(np.mean(errors)),
    }


def summarize_metrics(
    df: pd.DataFrame,
    peak_hours: set[int],
    eval_split: str,
) -> pd.DataFrame:
    eval_df = df.copy()
    if eval_split == "test":
        if "is_test" not in eval_df.columns:
            raise ValueError("--eval-split test requires an is_test column.")
        eval_df = eval_df[bool_series(eval_df["is_test"])].copy()
    if eval_df.empty:
        raise ValueError("No rows available for metric evaluation.")

    variants = {
        "raw_no_graph": "raw_no_graph",
        "global_calibrated": "global_calibrated",
        "hour_calibrated": "hour_calibrated",
        "zone_hour_calibrated": "zone_hour_calibrated",
        "peak_hour_calibrated": "peak_hour_calibrated",
    }
    hour = eval_df["target_hour"].dt.hour
    slices = {
        "all": pd.Series(True, index=eval_df.index),
        "peak_hours": hour.isin(peak_hours),
        "non_peak_hours": ~hour.isin(peak_hours),
        "hour_7_8": hour.isin({7, 8}),
        "morning_6_9": hour.isin({6, 7, 8, 9}),
    }

    rows = []
    for slice_name, mask in slices.items():
        part = eval_df[mask].copy()
        for variant, column in variants.items():
            rows.append(metric_row(part, variant=variant, slice_name=slice_name, pred_column=column))
    return pd.DataFrame(rows)


def write_calibrated_detail(df: pd.DataFrame, output_dir: Path, variant_column: str) -> Path:
    output = df.copy()
    if "base_pred" in output.columns:
        output["raw_base_pred"] = output["base_pred"]
        output["base_pred"] = output[variant_column]
    output["raw_refined_pred"] = (
        output["refined_pred"] if "refined_pred" in output.columns else output["raw_prediction"]
    )
    output["refined_pred"] = output[variant_column]
    output["residual_pred"] = 0.0
    output["residual_target"] = output["true_value"] - output[variant_column]
    output["calibration_variant"] = variant_column
    output["calibration_factor"] = np.where(
        output["raw_prediction"].to_numpy(dtype=float) > 1e-9,
        output[variant_column].to_numpy(dtype=float) / output["raw_prediction"].to_numpy(dtype=float),
        1.0,
    )
    path = output_dir / f"no_graph_{variant_column}_detailed.csv"
    output.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    peak_hours = set(int(hour) for hour in args.peak_hours)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_no_graph_predictions(args.detailed, args.prediction_column)
    calibrated_df = apply_expanding_calibration(
        raw_df,
        min_count=args.min_count,
        smoothing=args.smoothing,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
        peak_hours=peak_hours,
    )
    summary_df = summarize_metrics(
        calibrated_df,
        peak_hours=peak_hours,
        eval_split=args.eval_split,
    )
    summary_path = args.output_dir / "peak_tcn_calibration_summary.csv"
    predictions_path = args.output_dir / "peak_tcn_calibration_predictions.csv"
    summary_df.to_csv(summary_path, index=False)
    calibrated_df.to_csv(predictions_path, index=False)

    detail_paths = [
        write_calibrated_detail(calibrated_df, args.output_dir, "hour_calibrated"),
        write_calibrated_detail(calibrated_df, args.output_dir, "zone_hour_calibrated"),
        write_calibrated_detail(calibrated_df, args.output_dir, "peak_hour_calibrated"),
    ]

    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved predictions to {predictions_path}")
    for path in detail_paths:
        print(f"Saved from_ex5-compatible detail to {path}")


if __name__ == "__main__":
    main()
