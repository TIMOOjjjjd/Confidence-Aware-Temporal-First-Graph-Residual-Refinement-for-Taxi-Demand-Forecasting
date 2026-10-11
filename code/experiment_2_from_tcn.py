"""Run Experiment 2 using cached raw no-graph TCN predictions.

This script skips TCN training and prediction entirely. For each target hour it
loads cached no-graph TCN detailed predictions. The cached base_pred becomes
Experiment 2's temporal baseline, then the usual Experiment 2 G0/G1/G2/G3
node feature combinations are trained on top.
"""

from __future__ import annotations

import argparse
from revision_protocol import (
    add_revision_args, start_revision_run, freeze_cohort,
    annotate_predictions, finish_revision_run,
)
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from confidence_softmax import combine_confidence_components
from experiment_2_graph_refinement import (
    BASE_DIR,
    DEFAULT_CONFIDENCE_WEIGHTS,
    DEFAULT_DATA_PATH,
    DEFAULT_EDGE_WEIGHT_MATRIX,
    DEFAULT_LOOKUP_PATH,
    DEFAULT_RESULTS_DIR,
    EXCLUDED_ZONES,
    FEATURE_COLUMNS_BY_MODEL,
    GNN_MODEL_ORDER,
    GNNTrainingConfig,
    GraphContext,
    HISTORY_FEATURES,
    MODEL_ORDER,
    ROLLING_STEPS,
    START_TARGET,
    SplitMasks,
    build_gnn_features_for_model,
    build_graph_context,
    build_historical_gnn_training_data,
    build_zone_hourly_counts,
    collect_errors,
    compute_history_consistency_scores,
    compute_history_means,
    compute_prior_scores_from_zone_hourly_counts,
    evaluate_predictions,
    load_required_data,
    make_location_split_sets,
    masks_from_location_splits,
    metrics_record,
    normalize_window_starts,
    print_summary_table,
    save_results,
    select_zones,
    set_random_seed,
    summarize_results,
    train_residual_gnn,
    validate_matching_nodes,
    validate_windows,
)
from rolling_od_graph_utils import (
    add_rolling_od_args,
    build_rolling_od_graph_context,
    load_od_location_lookup,
    load_od_zone_names,
    uses_rolling_od_graph,
    validate_rolling_od_args,
)
from experiment_4_confidence_weight_optimization import (
    random_simplex_candidates,
    run_weight_search_mode,
)
from experiment_3_confidence_weighted_gnn_ablation import (
    train_residual_gnn_with_weighted_loss,
)
from tcn_feature_utils import (
    add_residual_training_split_arg,
    add_peak_shrinkage_args,
    select_residual_training_history,
    shrink_residual_predictions,
    validate_peak_shrinkage_args,
)


DEFAULT_TCN_DETAIL = (
    BASE_DIR
    / "ex5_weight_csv"
    / "results"
    / "experiment_5_graph_structure_comparison_detailed.csv"
)

# DEFAULT_TCN_DETAIL = (
#     BASE_DIR
#     / "results"
#     / "peak_tcn_calibration"
#     / "no_graph_hour_calibrated_detailed.csv"
# )

DEFAULT_PEAK_HOURS = [6, 7, 8, 9, 16, 17, 18]
DEFAULT_MORNING_PEAK_HOURS = [6, 7, 8, 9]
DEFAULT_OD_LOOKBACK_DAYS = 60
DEFAULT_OUTPUT_DIR = (
    DEFAULT_RESULTS_DIR / "revision_20260917" / "ex2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 2 residual GNN variants using cached no-graph TCN "
            "predictions instead of running the TCN baseline."
        )
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
        help="Detailed CSV containing cached no_graph TCN predictions.",
    )
    parser.add_argument(
        "--prediction-column",
        choices=["refined_pred", "base_pred"],
        default="base_pred",
        help="Prediction column to reuse from the cached TCN detail CSV.",
    )
    parser.add_argument(
        "--peak-hours",
        "--morning-peak-hours",
        dest="peak_hours",
        type=int,
        nargs="*",
        default=DEFAULT_PEAK_HOURS,
        help=(
            "Hours treated as peak for residual-training partitioning. These "
            "hours are excluded when --summary-hours or --history-hours is non_peak."
        ),
    )
    parser.add_argument(
        "--summary-hours",
        choices=["non_peak", "all"],
        default="all",
        help="Hours included in final summary metrics. Default includes all hours.",
    )
    parser.add_argument(
        "--history-hours",
        choices=["non_peak", "all"],
        default="all",
        help="Historical snapshots used for residual GNN training. Default includes all hours.",
    )
    parser.add_argument(
        "--graph-type",
        choices=["od"],
        default="od",
        help="Only OD graph mode is supported because residual graph mode needs TCN predictions.",
    )
    add_residual_training_split_arg(parser)
    add_rolling_od_args(parser)
    parser.set_defaults(od_lookback_days=DEFAULT_OD_LOOKBACK_DAYS)
    add_peak_shrinkage_args(parser)
    parser.set_defaults(peak_shrinkage_alpha=1.0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-target", type=pd.Timestamp, default=START_TARGET)
    parser.add_argument(
        "--window-starts",
        nargs="*",
        default=None,
        help=(
            "Optional explicit 24-hour window starts. Defaults to the same "
            "Experiment 2 windows."
        ),
    )
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
    parser.add_argument(
        "--confidence-weight-mode",
        choices=["uniform", "random_search", "learned_softmax"],
        default="uniform",
        help="Confidence component weighting used by each Experiment 2 residual GNN.",
    )
    parser.add_argument(
        "--random-candidates",
        type=int,
        default=24,
        help="Number of Dirichlet confidence-weight candidates for random_search.",
    )
    parser.add_argument(
        "--max-zones",
        type=int,
        default=None,
        help="Optional smoke-test limit. Default evaluates all non-excluded zones.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing experiment_2_graph_refinement_metrics.png.",
    )
    add_revision_args(parser)
    args = parser.parse_args()
    if not args.tcn_detail.exists():
        parser.error(f"--tcn-detail does not exist: {args.tcn_detail}")
    if not args.peak_hours:
        parser.error("--peak-hours must contain at least one hour.")
    invalid_hours = [hour for hour in args.peak_hours if hour < 0 or hour > 23]
    if invalid_hours:
        parser.error(f"Invalid peak hour(s): {invalid_hours}")
    if args.random_candidates <= 0:
        parser.error("--random-candidates must be positive.")
    validate_peak_shrinkage_args(parser, args)
    validate_rolling_od_args(parser, args)
    return args


def resolve_window_starts(args: argparse.Namespace) -> List[pd.Timestamp]:
    if args.window_starts is not None:
        return normalize_window_starts(args.window_starts)
    if pd.Timestamp(args.start_target) != START_TARGET:
        return [pd.Timestamp(args.start_target).floor("h")]
    return normalize_window_starts(None)


def load_no_graph_tcn_baseline(
    detail_path: Path,
    prediction_column: str,
) -> pd.DataFrame:
    columns = pd.read_csv(detail_path, nrows=0).columns.tolist()
    required = {
        "target_hour",
        "graph_type",
        "PULocationID",
        "true_value",
        prediction_column,
    }
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"{detail_path} is missing required column(s): {missing}")

    optional = [
        "variance", "mc_std",
        "prior_score",
        "stability_score",
        "history_consistency_score",
        "full_confidence",
    ]
    usecols = list(required) + [column for column in optional if column in columns]
    detail = pd.read_csv(detail_path, usecols=usecols, parse_dates=["target_hour"])
    detail = detail[detail["graph_type"].eq("no_graph")].copy()
    if detail.empty:
        raise ValueError("TCN detail CSV has no no_graph rows.")

    selected = detail.copy()
    selected = selected.rename(columns={prediction_column: "base_pred"})
    selected["selected_tcn_source"] = "no_graph"
    selected = selected.drop(columns=["graph_type"])

    keys = ["target_hour", "PULocationID"]
    duplicate_mask = selected.duplicated(subset=keys, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        values = [column for column in selected.columns if column not in keys]
        inconsistent = selected.groupby(keys)[values].nunique(dropna=False).gt(1).any(axis=1)
        if inconsistent.any():
            raise ValueError("Conflicting duplicate TCN cache rows; select a single run.")
        print(f"[info] collapsing {duplicate_count} identical cache rows", flush=True)
        selected = selected.drop_duplicates(subset=keys, keep="first")

    selected["PULocationID"] = selected["PULocationID"].astype(int)
    return selected.set_index(keys).sort_index()


def build_step_df_from_tcn(
    baseline_cache: pd.DataFrame,
    target_hour: pd.Timestamp,
    zones: Sequence[int],
    zone_hourly_counts: pd.Series,
    prior_scores: Dict[int, float],
) -> pd.DataFrame:
    target_hour = pd.Timestamp(target_hour)
    records: List[Dict[str, object]] = []

    try:
        hour_cache = baseline_cache.xs(target_hour, level="target_hour")
    except KeyError:
        hour_cache = pd.DataFrame()

    missing_zones: List[int] = []
    for zone_id in zones:
        zone_int = int(zone_id)
        history_means = compute_history_means(
            zone_hourly_counts,
            zone_int,
            target_hour,
        )
        if zone_int not in hour_cache.index:
            missing_zones.append(zone_int)
            records.append(
                {
                    "target_hour": target_hour,
                    "PULocationID": zone_int,
                    "base_pred": np.nan,
                    "true_value": np.nan,
                    "variance": np.nan,
                    "error": "missing no_graph tcn baseline",
                    "selected_tcn_source": "",
                    "full_confidence": np.nan,
                    **history_means,
                }
            )
            continue

        row = hour_cache.loc[zone_int]
        prior = float(row.get("prior_score", prior_scores.get(zone_int, 0.4)))
        stability = float(row.get("stability_score", 1.0))
        history_score = float(row.get("history_consistency_score", np.nan))
        confidence = float(row.get("full_confidence", np.nan))
        records.append(
            {
                "target_hour": target_hour,
                "PULocationID": zone_int,
                "base_pred": float(row["base_pred"]),
                "true_value": float(row["true_value"]),
                "variance": float(row.get("variance", float(row.get("mc_std", np.nan)) ** 2)),
                "error": "",
                "selected_tcn_source": row.get("selected_tcn_source", ""),
                "prior_score": prior,
                "stability_score": stability,
                "history_consistency_score": history_score,
                "confidence": confidence,
                "full_confidence": confidence,
                **history_means,
            }
        )

    if missing_zones:
        print(
            f"[{target_hour}] missing no_graph TCN baseline for "
            f"{len(missing_zones)} zone(s); those rows will be ignored by GNN data prep.",
            flush=True,
        )

    step_df = pd.DataFrame(records)
    for column in [
        "prior_score",
        "stability_score",
        "history_consistency_score",
        "confidence",
        "full_confidence",
    ]:
        if column not in step_df.columns:
            step_df[column] = np.nan
    fallback_history = compute_history_consistency_scores(step_df)
    step_df["prior_score"] = step_df["prior_score"].fillna(
        step_df["PULocationID"].map(lambda zone_id: prior_scores.get(int(zone_id), 0.4))
    )
    step_df["stability_score"] = step_df["stability_score"].fillna(1.0)
    step_df["history_consistency_score"] = step_df["history_consistency_score"].fillna(
        step_df["PULocationID"].map(lambda zone_id: fallback_history.get(int(zone_id), 0.6))
    )
    missing_confidence = step_df["confidence"].isna()
    if missing_confidence.any():
        step_df.loc[missing_confidence, "confidence"] = step_df.loc[
            missing_confidence
        ].apply(
            lambda row: combine_confidence_components(
                prior=float(row["prior_score"]),
                stability=float(row["stability_score"]),
                history_consistency=float(row["history_consistency_score"]),
                weights=DEFAULT_CONFIDENCE_WEIGHTS,
            )
            if np.isfinite(float(row["base_pred"]))
            else 0.0,
            axis=1,
        )
    step_df["full_confidence"] = step_df["full_confidence"].fillna(step_df["confidence"])
    step_df["confidence"] = step_df["confidence"].fillna(step_df["full_confidence"])
    return step_df


def train_ex2_residual_gnn(
    train_data: Optional[Data],
    train_splits: Optional[SplitMasks],
    inference_data: Data,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
    confidence_weight_mode: str,
    random_candidates: int,
    search_seed: int = 42,
):
    if confidence_weight_mode == "uniform":
        result = train_residual_gnn_with_weighted_loss(
            train_data=train_data, train_splits=train_splits,
            inference_data=inference_data, mode="none", device=device, cfg=cfg, seed=seed,
        )
        result.confidence_weights = np.full(3, np.nan, dtype=np.float32)
        return result, {"MAE": np.nan, "RMSE": np.nan, "MSE": np.nan}
    if confidence_weight_mode == "learned_softmax":
        result = train_residual_gnn(
            train_data=train_data,
            train_splits=train_splits,
            inference_data=inference_data,
            device=device,
            cfg=cfg,
            seed=seed,
        )
        val_metrics = {"MAE": float("nan"), "RMSE": float("nan"), "MSE": float("nan")}
        return result, val_metrics

    if confidence_weight_mode != "random_search":
        raise ValueError(f"Unsupported confidence weight mode: {confidence_weight_mode}")

    if (
        train_data is None
        or train_splits is None
        or int(train_splits.train.sum()) == 0
        or int(train_splits.val.sum()) == 0
    ):
        result = train_residual_gnn(
            train_data=train_data,
            train_splits=train_splits,
            inference_data=inference_data,
            device=device,
            cfg=cfg,
            seed=seed,
        )
        val_metrics = {"MAE": float("nan"), "RMSE": float("nan"), "MSE": float("nan")}
        return result, val_metrics

    candidates = random_simplex_candidates(count=random_candidates, seed=search_seed)
    result, _, selected_weights, val_metrics = run_weight_search_mode(
        mode="random_search",
        candidates=candidates,
        train_data=train_data,
        train_splits=train_splits,
        inference_data=inference_data,
        device=device,
        cfg=cfg,
        seed=seed,
    )
    result.confidence_weights = selected_weights
    return result, val_metrics


def run_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    print("Prediction column:", args.prediction_column)
    print("Baseline graph_type: no_graph")
    print("Peak hours:", sorted(peak_hours))
    print("Summary hours:", args.summary_hours)
    print("History hours:", args.history_hours)
    print("Residual training split:", args.residual_training_split)
    print("Confidence weight mode:", args.confidence_weight_mode)
    if args.confidence_weight_mode == "random_search":
        print("Random confidence candidates:", args.random_candidates)
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
    print("Window starts:", ", ".join(str(start) for start in window_starts))

    rolling_od_cache: Dict[pd.Timestamp, GraphContext] = {}
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

    zones = select_zones(df, args.max_zones)
    zones = [zone for zone in zones
             if graph_template.zone_to_location.get(graph_template.location_to_zone.get(zone)) == zone]
    zones = freeze_cohort(args, zones, baseline_cache, window_starts)
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

    detailed_records: List[Dict[str, object]] = []
    node_prediction_frames: List[pd.DataFrame] = []
    error_store: Dict[str, List[np.ndarray]] = {model_id: [] for model_id in MODEL_ORDER}

    for window_idx, window_start in enumerate(window_starts, start=1):
        historical_step_frames: List[pd.DataFrame] = []
        print(
            f"\n===== Experiment 2 from TCN no_graph window {window_idx}/{len(window_starts)} "
            f"start={window_start} ====="
        )
        for step in range(args.rolling_steps):
            global_step = (window_idx - 1) * args.rolling_steps + step
            target_hour = window_start + pd.Timedelta(hours=step)
            is_peak = int(target_hour.hour) in peak_hours
            include_in_summary = args.summary_hours == "all" or not is_peak
            include_in_history = args.history_hours == "all" or not is_peak
            print(
                f"\n///// target hour: {target_hour} "
                f"window {window_idx} step {step} /////"
            )
            if not include_in_summary and not include_in_history:
                print(
                    f"[{target_hour}] skipped peak hour "
                    "(not used for summary or history)",
                    flush=True,
                )
                continue

            set_random_seed(args.seed)
            if rolling_od:
                if target_hour not in rolling_od_cache:
                    print(f"[{target_hour}] start rolling OD graph", flush=True)
                    rolling_od_cache[target_hour] = build_rolling_od_graph_context(
                        args=args,
                        df=df,
                        lookup_df=lookup_df,
                        zone_names=od_zone_names,
                        location_to_zone=od_location_to_zone,
                        target_hour=target_hour,
                        build_graph_context_fn=build_graph_context,
                    )
                    print(f"[{target_hour}] done rolling OD graph", flush=True)
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

            source_counts = step_df["selected_tcn_source"].value_counts().to_dict()
            print(f"[{target_hour}] no_graph baseline source counts: {source_counts}")
            if not include_in_summary:
                if include_in_history:
                    historical_step_frames.append(step_df.copy())
                    print(
                        f"[{target_hour}] added to history only; skipped summary metrics",
                        flush=True,
                    )
                continue
            print(f"[{target_hour}] start feature and split prep", flush=True)
            inference_data_by_model = {
                model_id: build_gnn_features_for_model(model_id, step_df, graph_template)
                for model_id in GNN_MODEL_ORDER
            }
            reference_data = inference_data_by_model[GNN_MODEL_ORDER[0]]
            validate_matching_nodes(reference_data, inference_data_by_model)

            split_sets = make_location_split_sets(
                location_ids=zones,
                seed=args.split_seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            splits = masks_from_location_splits(
                data=reference_data,
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
            print(
                f"[{target_hour}] done feature and split prep; "
                f"history_snapshots={history_diag['history_snapshots']} "
                f"partition={history_diag['training_history_partition']} "
                f"all={history_diag['all_history_snapshots']} "
                f"peak={history_diag['peak_history_snapshots']} "
                f"non_peak={history_diag['non_peak_history_snapshots']}",
                flush=True,
            )

            train_data_by_model: Dict[str, Optional[Data]] = {}
            train_splits_by_model: Dict[str, Optional[SplitMasks]] = {}
            for model_id in GNN_MODEL_ORDER:
                print(f"[{target_hour}] build historical data for {model_id}", flush=True)
                train_data = build_historical_gnn_training_data(
                    history_frames=history_frames,
                    graph=graph_template,
                    feature_columns=FEATURE_COLUMNS_BY_MODEL[model_id],
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
                        print(f"[{target_hour}] {model_id} historical GNN training skipped: {exc}")
                        train_data = None
                train_data_by_model[model_id] = train_data
                train_splits_by_model[model_id] = train_splits

            test_mask_np = splits.test.cpu().numpy().astype(bool)
            y_true = reference_data.true_value.cpu().numpy()[test_mask_np]
            base_pred = reference_data.base_pred.cpu().numpy()[test_mask_np]
            prediction_template = pd.DataFrame({
                "target_hour": target_hour,
                "PULocationID": np.asarray(reference_data.location_ids)[test_mask_np],
                "is_test": True, "split": "test", "true_value": y_true,
                "base_pred": base_pred, "window_id": window_idx,
                "window_start": window_start, "window_step": step,
            })
            temporal_rows = prediction_template.copy()
            temporal_rows["model_id"] = "T4"
            temporal_rows["residual_pred"] = temporal_rows["raw_residual_pred"] = 0.0
            temporal_rows["refined_pred"] = temporal_rows["raw_refined_pred"] = base_pred
            temporal_rows["best_epoch"] = 0
            temporal_rows["shrinkage_alpha"] = 1.0
            temporal_rows["history_snapshots"] = history_diag["history_snapshots"]
            node_prediction_frames.append(annotate_predictions(temporal_rows, args, False))

            metrics_t4 = evaluate_predictions(y_true=y_true, y_pred=base_pred)
            collect_errors(error_store, "T4", y_true, base_pred)
            t4_record = metrics_record(
                target_hour=target_hour,
                model_id="T4",
                metrics=metrics_t4,
                splits=splits,
                node_count=reference_data.num_nodes,
                window_id=window_idx,
                window_start=window_start,
                window_step=step,
            )
            t4_record["shrinkage_alpha"] = np.nan
            t4_record["confidence_weight_mode"] = "none"
            t4_record["val_mae"] = np.nan
            t4_record["val_rmse"] = np.nan
            t4_record["val_mse"] = np.nan
            t4_record.update(history_diag)
            detailed_records.append(t4_record)

            hour_messages = [f"T4 MAE={metrics_t4['MAE']:.4f}"]
            for model_id in GNN_MODEL_ORDER:
                print(f"[{target_hour}] start model {model_id}", flush=True)
                result, val_metrics = train_ex2_residual_gnn(
                    train_data=train_data_by_model[model_id],
                    train_splits=train_splits_by_model[model_id],
                    inference_data=inference_data_by_model[model_id],
                    device=device,
                    cfg=gnn_cfg,
                    seed=args.seed,
                    confidence_weight_mode=args.confidence_weight_mode,
                    random_candidates=args.random_candidates,
                    search_seed=args.search_seed,
                )
                residual = result.residual_pred[test_mask_np]
                refined, shrinkage_alpha = shrink_residual_predictions(
                    base_pred=base_pred,
                    residual_pred=residual,
                    target_hour=target_hour,
                    shrinkage_hours=args.shrinkage_hours,
                    peak_alpha=args.peak_shrinkage_alpha,
                )
                metrics = evaluate_predictions(y_true=y_true, y_pred=refined)
                collect_errors(error_store, model_id, y_true, refined)
                gnn_record = metrics_record(
                    target_hour=target_hour,
                    model_id=model_id,
                    metrics=metrics,
                    splits=splits,
                    node_count=inference_data_by_model[model_id].num_nodes,
                    window_id=window_idx,
                    window_start=window_start,
                    window_step=step,
                    train_splits=train_splits_by_model[model_id],
                    best_epoch=result.best_epoch,
                    best_val_loss=result.best_val_loss,
                    confidence_weights=result.confidence_weights,
                )
                gnn_record["shrinkage_alpha"] = shrinkage_alpha
                gnn_record["confidence_weight_mode"] = args.confidence_weight_mode
                gnn_record["val_mae"] = val_metrics["MAE"]
                gnn_record["val_rmse"] = val_metrics["RMSE"]
                gnn_record["val_mse"] = val_metrics["MSE"]
                gnn_record.update(history_diag)
                detailed_records.append(gnn_record)
                node_rows = prediction_template.copy()
                node_rows["model_id"] = model_id
                node_rows["raw_residual_pred"] = residual
                node_rows["raw_refined_pred"] = base_pred + residual
                node_rows["residual_pred"] = residual * shrinkage_alpha
                node_rows["refined_pred"] = refined
                node_rows["shrinkage_alpha"] = shrinkage_alpha
                node_rows["best_epoch"] = result.best_epoch
                node_rows["history_snapshots"] = history_diag["history_snapshots"]
                node_prediction_frames.append(annotate_predictions(node_rows, args))

                weights = result.confidence_weights
                hour_messages.append(
                    f"{model_id} MAE={metrics['MAE']:.4f} "
                    f"(alpha={shrinkage_alpha:.2f}, "
                    f"val={val_metrics['MAE']:.4f}, "
                    f"w={weights[0]:.3f}/{weights[1]:.3f}/{weights[2]:.3f})"
                )
                print(
                    f"[{target_hour}] done model {model_id} MAE={metrics['MAE']:.4f}",
                    flush=True,
                )

            print(f"[{target_hour}] " + ", ".join(hour_messages), flush=True)
            if include_in_history:
                historical_step_frames.append(step_df.copy())

    detailed_df = pd.DataFrame(detailed_records)
    summary_df = summarize_results(error_store)
    pd.concat(node_prediction_frames, ignore_index=True).to_csv(
        args.results_dir / "experiment_2_node_predictions.csv", index=False
    )
    return summary_df, detailed_df


def main() -> None:
    args = parse_args()
    summary_df, detailed_df = run_experiment(args)
    save_results(
        summary_df=summary_df,
        detailed_df=detailed_df,
        results_dir=args.results_dir,
        make_plot=not args.no_plot,
    )
    finish_revision_run(args)
    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
