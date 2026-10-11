"""Matched forecasts, predeclared shrinkage sensitivity and paired block intervals.

Analysis only: no temporal or residual model training. Alpha grids are reported
as sensitivity analyses, NEVER selected by best test-set performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["model_seed", "target_hour", "PULocationID"]


def load_predictions(paths, variant_column=None):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=["target_hour"])
        if "is_test" in frame:
            test = frame.is_test.astype(str).str.lower().isin(["true", "1"])
            frame = frame.loc[test].copy()
        elif "split" in frame:
            frame = frame.loc[frame.split.eq("test")].copy()
        else:
            raise ValueError(f"No test mask in {path}")
        if "model_seed" not in frame:
            frame["model_seed"] = 42
        column = variant_column or next(
            (name for name in ["variant", "graph_type", "model_id", "mode"] if name in frame),
            None,
        )
        if column is None:
            raise ValueError(f"No variant identifier in {path}")
        frame["variant"] = frame[column].astype(str)
        # Legacy logs occasionally label temporal-only rows as using confidence.
        if "confidence_feature_used" in frame:
            if "graph_type" in frame:
                frame.loc[frame.graph_type.eq("no_graph"), "confidence_feature_used"] = False
            if "model_id" in frame:
                frame.loc[frame.model_id.eq("T4"), "confidence_feature_used"] = False
        if column != "variant" and "confidence_feature_used" in frame:
            frame["variant"] += "|confidence=" + frame.confidence_feature_used.astype(str)
        if column == "mode" and "residual_model" in frame:
            frame["variant"] = frame.residual_model + "|" + frame.variant
        if "raw_residual_pred" not in frame:
            if "shrinkage_alpha" in frame and not frame.shrinkage_alpha.fillna(1).eq(1).all():
                raise ValueError(f"Raw residuals unavailable for shrinkage analysis: {path}")
            frame["raw_residual_pred"] = frame.refined_pred - frame.base_pred
        frame["source_csv"] = str(Path(path).resolve())
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(KEYS + ["variant"]).any():
        raise ValueError("Duplicate seed/time/location/variant predictions; do not mix runs.")
    return combined


def match_predictions(frame, active_only=False):
    frame = frame.copy()
    if active_only:
        if "history_snapshots" not in frame:
            raise ValueError("Active-only analysis requires history_snapshots.")
        frame = frame.loc[frame.history_snapshots.gt(0)].copy()
    values = ["true_value", "base_pred", "refined_pred", "raw_residual_pred"]
    finite = np.isfinite(frame[values].to_numpy(dtype=float)).all(1)
    frame = frame.loc[finite].copy()
    variants = sorted(frame.variant.unique())
    counts = frame.groupby(KEYS).variant.nunique()
    common = counts[counts.eq(len(variants))].index
    matched = frame.set_index(KEYS).loc[common].reset_index()
    if matched.empty:
        raise ValueError("No common finite test predictions across all variants/seeds.")
    truth_spread = matched.groupby(KEYS).true_value.agg(lambda values: values.max()-values.min())
    base_spread = matched.groupby(KEYS).base_pred.agg(lambda values: values.max()-values.min())
    if truth_spread.gt(1e-5).any() or base_spread.gt(1e-5).any():
        raise ValueError("Models do not share identical truth and frozen TCN predictions.")
    coverage = frame.groupby("variant").size().rename("finite_test_rows").to_frame()
    coverage["matched_rows"] = matched.groupby("variant").size()
    coverage["excluded_rows"] = coverage.finite_test_rows - coverage.matched_rows
    temporal = matched.drop_duplicates(KEYS).copy()
    temporal["variant"] = "TCN"
    temporal["refined_pred"] = temporal.base_pred
    temporal["raw_residual_pred"] = 0.0
    matched = pd.concat([matched, temporal], ignore_index=True)
    return matched, coverage.reset_index()


def metric_record(frame, prediction):
    error = np.asarray(prediction) - frame.true_value.to_numpy()
    absolute = np.abs(error)
    return {"n": len(error), "MAE": absolute.mean(), "RMSE": np.sqrt(np.mean(error**2)),
            "P90_absolute_error": np.quantile(absolute, .90),
            "P95_absolute_error": np.quantile(absolute, .95)}


def paired_block_interval(frame, baseline="TCN", repeats=2000, seed=42):
    reference = frame.loc[frame.variant.eq(baseline)].set_index(KEYS)
    if reference.empty:
        raise ValueError(f"Paired baseline not present: {baseline}")
    results = []
    rng = np.random.default_rng(seed)
    for variant in sorted(set(frame.variant) - {baseline}):
        subset = frame.loc[frame.variant.eq(variant)].set_index(KEYS)
        paired = subset.join(reference[["refined_pred"]].rename(
            columns={"refined_pred": "baseline_pred"}), how="inner", validate="one_to_one")
        delta = np.abs(paired.refined_pred-paired.true_value) - np.abs(
            paired.baseline_pred-paired.true_value)
        blocks = pd.DataFrame({"delta": delta}).reset_index()
        blocks["block"] = blocks.target_hour.dt.floor("D")
        aggregates = blocks.groupby("block").delta.agg(["sum", "count"])
        n_blocks = len(aggregates)
        if n_blocks >= 2:
            draws = rng.integers(0, n_blocks, size=(repeats, n_blocks))
            sums, counts = aggregates["sum"].to_numpy(), aggregates["count"].to_numpy()
            bootstrap = sums[draws].sum(1) / counts[draws].sum(1)
            lower, upper = np.quantile(bootstrap, [.025, .975])
        else:
            lower = upper = np.nan
        results.append({"variant": variant, "baseline": baseline,
                        "MAE_delta": delta.mean(), "CI95_low": lower, "CI95_high": upper,
                        "day_blocks": n_blocks,
                        "scope": "paired day blocks; fixed observed training seeds"})
    return pd.DataFrame(results)


def analyze(frame, alphas, peak_hours, repeats=2000, baseline="TCN"):
    metrics, seed_metrics = [], []
    for alpha in alphas:
        rows = frame.copy()
        factor = np.where(rows.target_hour.dt.hour.isin(peak_hours), alpha, 1.0)
        rows["refined_pred"] = rows.base_pred + factor * rows.raw_residual_pred
        for (variant, seed), group in rows.groupby(["variant", "model_seed"]):
            seed_metrics.append({"variant": variant, "model_seed": seed, "peak_alpha": alpha,
                                 **metric_record(group, group.refined_pred)})
        for variant, group in rows.groupby("variant"):
            metrics.append({"variant": variant, "peak_alpha": alpha,
                            **metric_record(group, group.refined_pred)})
    per_seed = pd.DataFrame(seed_metrics)
    aggregate = per_seed.groupby(["variant", "peak_alpha"]).agg(
        seed_count=("model_seed", "nunique"), MAE_mean=("MAE", "mean"),
        MAE_seed_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"),
        RMSE_seed_std=("RMSE", "std"),
    ).reset_index()
    return (pd.DataFrame(metrics), per_seed, aggregate,
            paired_block_interval(frame, baseline=baseline, repeats=repeats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant-column", default=None)
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0., .3, .5, .7, 1.])
    parser.add_argument("--peak-hours", type=int, nargs="+", default=[6, 7, 8, 9, 16, 17, 18])
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--paired-baseline", default="TCN",
                        help="Exact variant identifier, e.g. residual_mlp|confidence=True.")
    args = parser.parse_args()
    if args.bootstrap_repeats < 1 or any(not np.isfinite(alpha) or alpha < 0
                                        for alpha in args.alphas):
        parser.error("Positive bootstrap count and nonnegative alphas required.")
    if any(hour < 0 or hour > 23 for hour in args.peak_hours):
        parser.error("Peak hours must be integers in [0, 23].")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Analysis output directory must be new.")
    frame = load_predictions(args.predictions, args.variant_column)
    matched, coverage = match_predictions(frame, args.active_only)
    metrics, seeds, summary, paired = analyze(
        matched, args.alphas, args.peak_hours, args.bootstrap_repeats,
        baseline=args.paired_baseline,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, output in [("coverage", coverage), ("sensitivity", metrics),
                         ("per_seed", seeds), ("seed_summary", summary), ("paired_ci", paired)]:
        output.to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps({
        "sources": [str(path.resolve()) for path in args.predictions],
        "active_only": args.active_only, "alphas": args.alphas,
        "peak_hours": args.peak_hours, "bootstrap_repeats": args.bootstrap_repeats,
        "paired_baseline": args.paired_baseline,
        "paired_interval_predictions": "original saved forecasts, not alpha sensitivity rows",
        "alpha_selection": "none; predeclared sensitivity, not test-selected tuning",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
