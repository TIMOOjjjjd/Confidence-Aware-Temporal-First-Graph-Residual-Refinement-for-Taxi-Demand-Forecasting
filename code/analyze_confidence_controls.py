"""Confidence ranking with strictly historical demand-scale controls, not calibration."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_revision_data import load_cached_trip_data
from experiment_3_confidence_weighted_gnn_ablation import build_zone_hourly_counts


def historical_scale(frame, counts, hours=168):
    start = frame.target_hour.min() - pd.Timedelta(hours=hours)
    end = frame.target_hour.max()
    zones = sorted(frame.PULocationID.unique())
    index = pd.date_range(start, end, freq="h")
    table = counts.unstack(0).reindex(index=index, columns=zones).fillna(0.)
    scale = table.rolling(hours, min_periods=hours).mean().shift(1)
    scale.index.name = "target_hour"
    scale.columns.name = "PULocationID"
    scales = scale.stack().dropna().rename("historical_mean").reset_index()
    requested = frame[["target_hour", "PULocationID"]].drop_duplicates()
    return requested.merge(scales, on=["target_hour", "PULocationID"],
                           how="left", validate="one_to_one")


def correlations(frame, score, error):
    valid = frame[[score, error]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid[score].nunique() < 2 or valid[error].nunique() < 2:
        return {"n": len(valid), "Pearson": np.nan, "Spearman": np.nan}
    return {"n": len(valid), "Pearson": valid[score].corr(valid[error]),
            "Spearman": valid[score].rank().corr(valid[error].rank())}


def analyze(frame, counts, bootstrap_repeats=1000, seed=42):
    scale = historical_scale(frame, counts)
    frame = frame.merge(scale, on=["target_hour", "PULocationID"],
                        how="left", validate="many_to_one")
    if frame.historical_mean.isna().any():
        raise ValueError("Historical demand scales missing for some predictions.")
    frame["absolute_error"] = (frame.refined_pred-frame.true_value).abs()
    frame["normalized_error"] = frame.absolute_error / frame.historical_mean.clip(lower=1.)
    frame["demand_stratum"] = pd.qcut(frame.historical_mean, 4, duplicates="drop").astype(str)
    scores = [name for name in ["historical_mean", "prior_score", "stability_score",
                               "history_consistency_score", "full_confidence", "sample_weight"]
              if name in frame and frame[name].notna().any()]
    if "variance" in frame and frame.variance.notna().any():
        frame["negative_variance"] = -frame.variance
        scores.append("negative_variance")
    rows = []
    for stratum, subset in [("all", frame)] + list(frame.groupby("demand_stratum")):
        for score in scores:
            for error in ["absolute_error", "normalized_error"]:
                rows.append({"demand_stratum": stratum, "score": score, "error": error,
                             **correlations(subset, score, error)})
    rng = np.random.default_rng(seed)
    frame["day"] = frame.target_hour.dt.floor("D")
    blocks = [group for _, group in frame.groupby("day")]
    interval_rows = []
    if len(blocks) >= 2:
        estimates = {(score, error): [] for score in scores
                     for error in ["absolute_error", "normalized_error"]}
        for _ in range(bootstrap_repeats):
            sample = pd.concat([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))])
            for (score, error), values in estimates.items():
                values.append(correlations(sample, score, error)["Spearman"])
        for (score, error), values in estimates.items():
            finite = np.asarray(values)[np.isfinite(values)]
            low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            interval_rows.append({"score": score, "error": error,
                                  "Spearman_CI95_low": low, "Spearman_CI95_high": high,
                                  "day_blocks": len(blocks)})
    return frame, pd.DataFrame(rows), pd.DataFrame(interval_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 1:
        parser.error("--bootstrap-repeats must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Confidence analysis output must be new.")
    frame = pd.read_csv(args.predictions, parse_dates=["target_hour"])
    frame = frame.loc[frame.is_test.astype(str).str.lower().isin(["true", "1"])].copy()
    if args.mode is not None:
        frame = frame.loc[frame["mode"].eq(args.mode)].copy()
    if frame.empty or frame.duplicated(["target_hour", "PULocationID"]).any():
        raise ValueError("Select exactly one fitted mode/seed on unique test samples.")
    counts = build_zone_hourly_counts(load_cached_trip_data(args.data))
    annotated, results, intervals = analyze(frame, counts, args.bootstrap_repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output_dir / "scaled_predictions.csv", index=False)
    results.to_csv(args.output_dir / "correlations.csv", index=False)
    intervals.to_csv(args.output_dir / "correlation_intervals.csv", index=False)
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps({
        "predictions": str(args.predictions.resolve()), "data": str(args.data.resolve()),
        "mode": args.mode, "bootstrap_repeats": args.bootstrap_repeats,
        "normalization": "absolute_error / max(pretarget 168h mean demand, 1)",
        "interpretation": "ranking association; not calibrated predictive uncertainty",
    }, indent=2), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
