"""Run Experiment 6 using cached raw no-graph TCN predictions.

The residual GNN uses base_pred, full_confidence, and the three historical mean
features as node features before validating confidence reliability.
"""

from __future__ import annotations

import argparse
from revision_protocol import (
    add_revision_args, start_revision_run, freeze_cohort,
    annotate_predictions, finish_revision_run,
)
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from experiment_2_from_tcn import (
    DEFAULT_PEAK_HOURS,
    DEFAULT_TCN_DETAIL,
    build_step_df_from_tcn,
    load_no_graph_tcn_baseline,
)
from experiment_3_confidence_weighted_gnn_ablation import (
    DEFAULT_DATA_PATH,
    DEFAULT_EDGE_WEIGHT_MATRIX,
    DEFAULT_LOOKUP_PATH,
    DEFAULT_RESULTS_DIR,
    EXCLUDED_ZONES,
    ROLLING_STEPS,
    START_TARGET,
    GNNTrainingConfig,
    SplitMasks,
    build_graph_context,
    build_zone_hourly_counts,
    compute_prior_scores_from_zone_hourly_counts,
    load_required_data,
    make_location_split_sets,
    masks_from_location_splits,
    normalize_window_starts,
    select_zones,
    set_random_seed,
    validate_windows,
)
from experiment_4_confidence_weight_optimization import (
    MODE_ORDER,
    run_single_mode,
)
from tcn_feature_utils import (
    add_residual_training_split_arg,
    add_peak_shrinkage_args,
    apply_peak_residual_shrinkage,
    build_gnn_features_with_history,
    build_historical_gnn_training_data_with_history,
    recompute_prediction_df_test_metrics,
    select_residual_training_history,
    validate_peak_shrinkage_args,
)
from experiment_6_confidence_reliability_validation import (
    compute_correlations,
    compute_quantile_errors,
    prepare_test_predictions,
    print_summary_table,
    save_results,
    summarize_confidence_groups,
)
from rolling_od_graph_utils import (
    add_rolling_od_args,
    build_rolling_od_graph_context,
    load_od_location_lookup,
    load_od_zone_names,
    uses_rolling_od_graph,
    validate_rolling_od_args,
)


DEFAULT_OUTPUT_DIR = (
    DEFAULT_RESULTS_DIR / "revision_20260917" / "ex6"
)
DEFAULT_OD_LOOKBACK_DAYS = 60
DEFAULT_CONFIDENCE_MODE = "random_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 6 using cached no_graph TCN predictions."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_WEIGHT_MATRIX)
    parser.add_argument(
        "--tcn-detail",
        "--ex5-detail",
        dest="tcn_detail",
        type=Path,
        default=DEFAULT_TCN_DETAIL,
    )
    parser.add_argument(
        "--prediction-column",
        choices=["refined_pred", "base_pred"],
        default="base_pred",
    )
    parser.add_argument(
        "--peak-hours",
        "--morning-peak-hours",
        dest="peak_hours",
        type=int,
        nargs="*",
        default=DEFAULT_PEAK_HOURS,
    )
    parser.add_argument("--summary-hours", choices=["non_peak", "all"], default="all")
    parser.add_argument("--history-hours", choices=["non_peak", "all"], default="all")
    parser.add_argument("--graph-type", choices=["od"], default="od")
    add_rolling_od_args(parser)
    parser.set_defaults(od_lookback_days=DEFAULT_OD_LOOKBACK_DAYS)
    add_residual_training_split_arg(parser)
    add_peak_shrinkage_args(parser)
    parser.set_defaults(peak_shrinkage_alpha=1.0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-target", type=pd.Timestamp, default=START_TARGET)
    parser.add_argument("--window-starts", nargs="*", default=None)
    parser.add_argument("--rolling-steps", type=int, default=ROLLING_STEPS)
    parser.add_argument("--excluded-zones", type=int, nargs="*", default=EXCLUDED_ZONES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--gnn-epochs", type=int, default=300)
    parser.add_argument("--gnn-hidden-dim", type=int, default=256)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)
    parser.add_argument("--gnn-lr", type=float, default=0.01)
    parser.add_argument("--gnn-patience", type=int, default=40)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--random-candidates", type=int, default=24)
    parser.add_argument("--learned-entropy", type=float, default=0.0)
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument("--mode", choices=MODE_ORDER, default=DEFAULT_CONFIDENCE_MODE)
    parser.add_argument("--confidence-column", default="sample_weight")
    parser.add_argument("--analysis-prediction-column", default="refined_pred")
    parser.add_argument("--truth-column", default="true_value")
    parser.add_argument("--max-zones", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--detailed-csv", type=Path, default=None,
                        help="Analysis only: reuse saved node predictions, no model training.")
    add_revision_args(parser)
    args = parser.parse_args()
    if not args.tcn_detail.exists():
        parser.error(f"--tcn-detail does not exist: {args.tcn_detail}")
    if args.quantile_bins < 2:
        parser.error("--quantile-bins must be at least 2.")
    if args.random_candidates <= 0:
        parser.error("--random-candidates must be positive.")
    if not args.peak_hours:
        parser.error("--peak-hours must contain at least one hour.")
    invalid_hours = [hour for hour in args.peak_hours if hour < 0 or hour > 23]
    if invalid_hours:
        parser.error(f"Invalid peak hour(s): {invalid_hours}")
    validate_peak_shrinkage_args(parser, args)
    validate_rolling_od_args(parser, args)
    return args


def resolve_window_starts(args: argparse.Namespace) -> List[pd.Timestamp]:
    if args.window_starts is not None:
        return normalize_window_starts(args.window_starts)
    if pd.Timestamp(args.start_target) != START_TARGET:
        return [pd.Timestamp(args.start_target).floor("h")]
    return normalize_window_starts(None)


def run_prediction_generation(args: argparse.Namespace) -> pd.DataFrame:
    start_revision_run(args, __file__)
    set_random_seed(args.seed)
    peak_hours = set(int(hour) for hour in args.peak_hours)
    baseline_cache = load_no_graph_tcn_baseline(
        detail_path=args.tcn_detail,
        prediction_column=args.prediction_column,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("TCN detail:", args.tcn_detail)
    print("Baseline graph_type: no_graph")
    print("Confidence mode:", args.mode)
    print("Peak hours:", sorted(peak_hours))
    print("Summary hours:", args.summary_hours)
    print("History hours:", args.history_hours)
    print("Residual training split:", args.residual_training_split)
    print("Shrinkage hours:", sorted(set(int(hour) for hour in args.shrinkage_hours)))
    print("Peak shrinkage alpha:", args.peak_shrinkage_alpha)

    rolling_od = uses_rolling_od_graph(args)
    df, lookup_df, graph_template = load_required_data(
        data_path=args.data,
        lookup_path=args.lookup,
        edge_csv=args.edge_csv,
        excluded_zones=args.excluded_zones,
        include_dropoff=rolling_od,
    )
    window_starts = resolve_window_starts(args)
    validate_windows(df, window_starts, args.rolling_steps)

    rolling_od_cache: Dict[pd.Timestamp, object] = {}
    od_zone_names: List[str] = []
    od_location_to_zone: Dict[int, str] = {}
    if rolling_od:
        od_zone_names = load_od_zone_names(args)
        od_location_to_zone = load_od_location_lookup(args)
        first_target_hour = window_starts[0]
        rolling_od_cache[first_target_hour] = build_rolling_od_graph_context(
            args=args,
            df=df,
            lookup_df=lookup_df,
            zone_names=od_zone_names,
            location_to_zone=od_location_to_zone,
            target_hour=first_target_hour,
            build_graph_context_fn=build_graph_context,
        )
        graph_template = rolling_od_cache[first_target_hour]

    zones = select_zones(df, graph_template, args.max_zones)
    zones = [zone for zone in zones if zone in graph_template.zone_to_location.values()]
    zones = freeze_cohort(args, zones, baseline_cache, window_starts)
    print("Graph-compatible zones selected:", len(zones))
    zone_hourly_counts = build_zone_hourly_counts(df)
    gnn_cfg = GNNTrainingConfig(
        hidden_dim=args.gnn_hidden_dim,
        dropout=args.gnn_dropout,
        learning_rate=args.gnn_lr,
        epochs=args.gnn_epochs,
        patience=args.gnn_patience,
        model_type=args.residual_model,
        use_confidence_feature=not args.exclude_confidence_feature,
    )

    detailed_frames: List[pd.DataFrame] = []
    for window_idx, window_start in enumerate(window_starts, start=1):
        historical_step_frames: List[pd.DataFrame] = []
        print(
            f"\n===== Experiment 6 from TCN no_graph window {window_idx}/{len(window_starts)} "
            f"start={window_start} ====="
        )
        for step in range(args.rolling_steps):
            global_step = (window_idx - 1) * args.rolling_steps + step
            target_hour = window_start + pd.Timedelta(hours=step)
            is_peak = int(target_hour.hour) in peak_hours
            include_in_summary = args.summary_hours == "all" or not is_peak
            include_in_history = args.history_hours == "all" or not is_peak
            if not include_in_summary and not include_in_history:
                print(f"[{target_hour}] skipped peak hour", flush=True)
                continue

            set_random_seed(args.seed)
            if include_in_summary and rolling_od:
                if target_hour not in rolling_od_cache:
                    rolling_od_cache[target_hour] = build_rolling_od_graph_context(
                        args=args,
                        df=df,
                        lookup_df=lookup_df,
                        zone_names=od_zone_names,
                        location_to_zone=od_location_to_zone,
                        target_hour=target_hour,
                        build_graph_context_fn=build_graph_context,
                    )
                graph_template = rolling_od_cache[target_hour]

            prior_scores = compute_prior_scores_from_zone_hourly_counts(
                zone_hourly_counts=zone_hourly_counts,
                target_hour=target_hour,
            )
            step_df = build_step_df_from_tcn(
                baseline_cache=baseline_cache,
                target_hour=target_hour,
                zones=zones,
                zone_hourly_counts=zone_hourly_counts,
                prior_scores=prior_scores,
            )
            step_df["window_id"] = window_idx
            step_df["window_start"] = window_start
            step_df["window_step"] = step

            if not include_in_summary:
                if include_in_history:
                    historical_step_frames.append(step_df.copy())
                continue

            inference_data = build_gnn_features_with_history(
                step_df=step_df,
                graph=graph_template,
                use_edge_weight=True,
            )
            split_sets = make_location_split_sets(
                location_ids=zones,
                seed=args.split_seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            inference_splits = masks_from_location_splits(
                data=inference_data,
                split_sets=split_sets,
                require_train_val=False,
                require_test=True,
            )
            history_frames, history_diag = select_residual_training_history(
                history_frames=historical_step_frames,
                target_hour=target_hour,
                peak_hours=peak_hours,
                split_mode=args.residual_training_split,
            )
            train_data = build_historical_gnn_training_data_with_history(
                history_frames=history_frames,
                graph=graph_template,
                use_edge_weight=True,
            )
            train_splits: Optional[SplitMasks] = None
            if train_data is not None:
                try:
                    train_splits = masks_from_location_splits(
                        data=train_data,
                        split_sets=split_sets,
                        require_train_val=True,
                        require_test=False,
                    )
                except ValueError as exc:
                    print(f"[{target_hour}] historical GNN training skipped: {exc}")
                    train_data = None

            prediction_df, hourly_record = run_single_mode(
                target_hour=target_hour,
                mode=args.mode,
                train_data=train_data,
                train_splits=train_splits,
                inference_data=inference_data,
                inference_splits=inference_splits,
                device=device,
                cfg=gnn_cfg,
                seed=args.seed,
                grid_step=args.grid_step,
                random_candidates=args.random_candidates,
                learned_entropy=args.learned_entropy,
                search_seed=args.search_seed,
            )
            prediction_df, shrinkage_alpha = apply_peak_residual_shrinkage(
                prediction_df=prediction_df,
                target_hour=target_hour,
                shrinkage_hours=args.shrinkage_hours,
                peak_alpha=args.peak_shrinkage_alpha,
            )
            metrics = recompute_prediction_df_test_metrics(prediction_df)
            prediction_df["window_id"] = window_idx
            prediction_df["window_start"] = window_start
            prediction_df["window_step"] = step
            prediction_df["hourly_mae"] = metrics["MAE"]
            prediction_df["hourly_rmse"] = metrics["RMSE"]
            prediction_df["hourly_mse"] = metrics["MSE"]
            for column, value in history_diag.items():
                prediction_df[column] = value
            detailed_frames.append(annotate_predictions(prediction_df, args))
            test_mean_confidence = prediction_df.loc[
                prediction_df["is_test"],
                "sample_weight",
            ].mean()
            print(
                f"[{target_hour}] test_MAE={metrics['MAE']:.4f} "
                f"alpha={shrinkage_alpha:.2f} "
                f"mean_test_confidence={test_mean_confidence:.4f}",
                flush=True,
            )
            if include_in_history:
                historical_step_frames.append(step_df.copy())

    return pd.concat(detailed_frames, ignore_index=True)


def run_experiment(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.detailed_csv is not None:
        start_revision_run(args, __file__)
        detailed_df = pd.read_csv(args.detailed_csv, parse_dates=["target_hour"])
    else:
        detailed_df = run_prediction_generation(args)
    test_predictions_df = prepare_test_predictions(
        predictions_df=detailed_df,
        mode=args.mode,
        confidence_column=args.confidence_column,
        prediction_column=args.analysis_prediction_column,
        truth_column=args.truth_column,
    )
    summary_df = summarize_confidence_groups(test_df=test_predictions_df)
    correlations_df = compute_correlations(test_df=test_predictions_df)
    quantile_df = compute_quantile_errors(
        test_df=test_predictions_df,
        quantile_bins=args.quantile_bins,
    )
    return summary_df, correlations_df, quantile_df, detailed_df, test_predictions_df


def main() -> None:
    args = parse_args()
    summary_df, correlations_df, quantile_df, detailed_df, test_predictions_df = run_experiment(
        args
    )
    save_results(
        summary_df=summary_df,
        correlations_df=correlations_df,
        quantile_df=quantile_df,
        detailed_df=detailed_df,
        test_predictions_df=test_predictions_df,
        results_dir=args.results_dir,
        make_plot=not args.no_plot,
    )
    finish_revision_run(args)
    print_summary_table(summary_df, correlations_df)


if __name__ == "__main__":
    main()
