"""Shared helpers for reruns that start from cached no-graph TCN predictions."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from experiment_3_confidence_weighted_gnn_ablation import (
    clamp_score,
    remap_edges_and_weights_to_valid_nodes,
)


HISTORY_FEATURE_COLUMNS = ("mean_24h", "mean_168h", "mean_720h")
DEFAULT_SHRINKAGE_HOURS = (6, 7, 8, 9, 16, 17, 18)
DEFAULT_PEAK_SHRINKAGE_ALPHA = 0.3


def add_residual_training_split_arg(parser) -> None:
    parser.add_argument(
        "--residual-training-split",
        choices=["all", "peak_non_peak"],
        default="peak_non_peak",
        help=(
            "Historical residual snapshots used for each GNN. 'peak_non_peak' "
            "trains peak target hours only on prior peak snapshots and non-peak "
            "target hours only on prior non-peak snapshots."
        ),
    )


def add_peak_shrinkage_args(parser) -> None:
    parser.add_argument(
        "--shrinkage-hours",
        type=int,
        nargs="*",
        default=list(DEFAULT_SHRINKAGE_HOURS),
        help="Clock hours where GNN residual predictions are multiplied by alpha.",
    )
    parser.add_argument(
        "--peak-shrinkage-alpha",
        type=float,
        default=DEFAULT_PEAK_SHRINKAGE_ALPHA,
        help=(
            "Residual multiplier for --shrinkage-hours. Use 1.0 to disable "
            "peak-hour shrinkage."
        ),
    )


def validate_peak_shrinkage_args(parser, args) -> None:
    invalid_hours = [hour for hour in args.shrinkage_hours if hour < 0 or hour > 23]
    if invalid_hours:
        parser.error(f"Invalid shrinkage hour(s): {invalid_hours}")
    if args.peak_shrinkage_alpha < 0.0 or args.peak_shrinkage_alpha > 1.0:
        parser.error("--peak-shrinkage-alpha must be between 0.0 and 1.0.")


def shrinkage_alpha_for_hour(
    target_hour: pd.Timestamp,
    shrinkage_hours: Iterable[int],
    peak_alpha: float,
) -> float:
    hours = {int(hour) for hour in shrinkage_hours}
    if int(pd.Timestamp(target_hour).hour) in hours:
        return float(peak_alpha)
    return 1.0


def shrink_residual_predictions(
    base_pred: np.ndarray,
    residual_pred: np.ndarray,
    target_hour: pd.Timestamp,
    shrinkage_hours: Iterable[int],
    peak_alpha: float,
) -> Tuple[np.ndarray, float]:
    alpha = shrinkage_alpha_for_hour(
        target_hour=target_hour,
        shrinkage_hours=shrinkage_hours,
        peak_alpha=peak_alpha,
    )
    refined = np.asarray(base_pred, dtype=float) + alpha * np.asarray(residual_pred, dtype=float)
    return refined, alpha


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"MAE": float("nan"), "RMSE": float("nan"), "MSE": float("nan")}
    errors = y_pred[mask] - y_true[mask]
    mse = float(np.mean(errors**2))
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
    }


def apply_peak_residual_shrinkage(
    prediction_df: pd.DataFrame,
    target_hour: pd.Timestamp,
    shrinkage_hours: Iterable[int],
    peak_alpha: float,
) -> Tuple[pd.DataFrame, float]:
    alpha = shrinkage_alpha_for_hour(
        target_hour=target_hour,
        shrinkage_hours=shrinkage_hours,
        peak_alpha=peak_alpha,
    )
    prediction_df = prediction_df.copy()
    if "raw_residual_pred" not in prediction_df.columns:
        prediction_df["raw_residual_pred"] = prediction_df["residual_pred"]
    if "raw_refined_pred" not in prediction_df.columns:
        prediction_df["raw_refined_pred"] = prediction_df["refined_pred"]
    prediction_df["shrinkage_alpha"] = alpha
    prediction_df["residual_pred"] = prediction_df["raw_residual_pred"].astype(float) * alpha
    prediction_df["refined_pred"] = (
        prediction_df["base_pred"].astype(float) + prediction_df["residual_pred"]
    )
    return prediction_df, alpha


def recompute_prediction_df_test_metrics(
    prediction_df: pd.DataFrame,
    prediction_column: str = "refined_pred",
) -> Dict[str, float]:
    test_df = prediction_df[prediction_df["is_test"]]
    return error_metrics(
        y_true=test_df["true_value"].to_numpy(dtype=float),
        y_pred=test_df[prediction_column].to_numpy(dtype=float),
    )


def history_frame_is_peak(frame: pd.DataFrame, peak_hours: set[int]) -> bool:
    if frame.empty or "target_hour" not in frame.columns:
        return False
    hours = pd.to_datetime(frame["target_hour"]).dt.hour.dropna().unique()
    return bool(len(hours) > 0 and int(hours[0]) in peak_hours)


def filter_history_frames_before(
    history_frames: Sequence[pd.DataFrame],
    target_hour: pd.Timestamp,
) -> List[pd.DataFrame]:
    filtered: List[pd.DataFrame] = []
    target_ts = pd.Timestamp(target_hour)
    for frame in history_frames:
        if frame.empty:
            continue
        if "target_hour" not in frame.columns:
            filtered.append(frame)
            continue
        frame_hours = pd.to_datetime(frame["target_hour"])
        if frame_hours.max() < target_ts:
            filtered.append(frame)
    return filtered


def select_residual_training_history(
    history_frames: Sequence[pd.DataFrame],
    target_hour: pd.Timestamp,
    peak_hours: set[int],
    split_mode: str,
) -> Tuple[List[pd.DataFrame], Dict[str, object]]:
    prior_frames = filter_history_frames_before(history_frames, target_hour)
    peak_frames = [frame for frame in prior_frames if history_frame_is_peak(frame, peak_hours)]
    non_peak_frames = [
        frame for frame in prior_frames if not history_frame_is_peak(frame, peak_hours)
    ]

    target_is_peak = int(pd.Timestamp(target_hour).hour) in peak_hours
    if split_mode == "peak_non_peak":
        selected_frames = peak_frames if target_is_peak else non_peak_frames
        partition = "peak" if target_is_peak else "non_peak"
    elif split_mode == "all":
        selected_frames = prior_frames
        partition = "all"
    else:
        raise ValueError(f"Unsupported residual training split: {split_mode}")

    diagnostics = {
        "residual_training_split": split_mode,
        "target_period": "peak" if target_is_peak else "non_peak",
        "training_history_partition": partition,
        "history_snapshots": len(selected_frames),
        "all_history_snapshots": len(prior_frames),
        "peak_history_snapshots": len(peak_frames),
        "non_peak_history_snapshots": len(non_peak_frames),
    }
    return selected_frames, diagnostics


def build_gnn_features_with_history(
    step_df: pd.DataFrame,
    graph,
    use_edge_weight: bool = True,
) -> Data:
    """Build residual-GNN data with base, confidence, and raw history features.

    Node feature order:
        base_pred, full_confidence, log1p(mean_24h), log1p(mean_168h), log1p(mean_720h)
    """

    node_count = len(graph.zone_names)
    node_pred = torch.full((node_count,), float("nan"), dtype=torch.float32)
    node_label = torch.full((node_count,), float("nan"), dtype=torch.float32)
    prior = torch.full((node_count,), 0.4, dtype=torch.float32)
    stability = torch.full((node_count,), 0.5, dtype=torch.float32)
    history_consistency = torch.full((node_count,), 0.6, dtype=torch.float32)
    full_confidence = torch.full((node_count,), 0.5, dtype=torch.float32)
    history_tensor = torch.zeros((node_count, len(HISTORY_FEATURE_COLUMNS)), dtype=torch.float32)
    location_id = torch.full((node_count,), -1, dtype=torch.long)

    for row in step_df.itertuples():
        loc_id = int(row.PULocationID)
        zone_name = graph.location_to_zone.get(loc_id)
        if zone_name is None:
            continue
        node_idx = graph.zone_idx_map.get(zone_name)
        if node_idx is None:
            continue

        node_pred[node_idx] = float(row.base_pred)
        node_label[node_idx] = float(row.true_value)
        location_id[node_idx] = loc_id
        prior[node_idx] = clamp_score(float(row.prior_score), default=0.4)
        stability[node_idx] = clamp_score(float(row.stability_score), default=0.5)
        history_consistency[node_idx] = clamp_score(
            float(row.history_consistency_score),
            default=0.6,
        )
        full_confidence[node_idx] = clamp_score(float(row.full_confidence), default=0.5)
        for feature_idx, feature_name in enumerate(HISTORY_FEATURE_COLUMNS):
            value = float(getattr(row, feature_name, np.nan))
            history_tensor[node_idx, feature_idx] = max(value, 0.0) if np.isfinite(value) else 0.0

    valid_indices = torch.where(~torch.isnan(node_pred) & ~torch.isnan(node_label))[0]
    if valid_indices.numel() < 3:
        raise ValueError("Not enough valid graph nodes to train/evaluate residual GNN.")

    edge_index, edge_weight = remap_edges_and_weights_to_valid_nodes(
        edge_index=graph.edge_index,
        valid_indices=valid_indices,
        edge_weight=graph.edge_weight if use_edge_weight else None,
    )
    node_pred = node_pred[valid_indices]
    node_label = node_label[valid_indices]
    prior = prior[valid_indices]
    stability = stability[valid_indices]
    history_consistency = history_consistency[valid_indices]
    full_confidence = full_confidence[valid_indices]
    history_tensor = history_tensor[valid_indices]
    location_id = location_id[valid_indices]

    x_feat = torch.cat(
        [
            node_pred.unsqueeze(1),
            full_confidence.unsqueeze(1),
            torch.log1p(history_tensor),
        ],
        dim=1,
    )

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        y=node_label - node_pred,
        base_pred=node_pred,
        true_value=node_label,
        location_id=location_id,
        prior_score=prior,
        stability_score=stability,
        history_consistency_score=history_consistency,
        full_confidence=full_confidence,
    )
    if edge_weight is not None:
        data.edge_weight = edge_weight
    data.confidence_feature_index = 1
    return data


def build_historical_gnn_training_data_with_history(
    history_frames: Sequence[pd.DataFrame],
    graph,
    use_edge_weight: bool = True,
) -> Optional[Data]:
    """Concatenate historical residual snapshots with the same history-rich x."""

    snapshots: List[Data] = []
    for frame in history_frames:
        if frame.empty:
            continue
        try:
            snapshots.append(
                build_gnn_features_with_history(
                    step_df=frame,
                    graph=graph,
                    use_edge_weight=use_edge_weight,
                )
            )
        except ValueError:
            continue

    if not snapshots:
        return None

    x_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    base_parts: List[torch.Tensor] = []
    true_parts: List[torch.Tensor] = []
    location_parts: List[torch.Tensor] = []
    prior_parts: List[torch.Tensor] = []
    stability_parts: List[torch.Tensor] = []
    history_parts: List[torch.Tensor] = []
    confidence_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []
    edge_weight_parts: List[torch.Tensor] = []

    offset = 0
    for snapshot in snapshots:
        x_parts.append(snapshot.x)
        y_parts.append(snapshot.y)
        base_parts.append(snapshot.base_pred)
        true_parts.append(snapshot.true_value)
        location_parts.append(snapshot.location_id)
        prior_parts.append(snapshot.prior_score)
        stability_parts.append(snapshot.stability_score)
        history_parts.append(snapshot.history_consistency_score)
        confidence_parts.append(snapshot.full_confidence)

        if snapshot.edge_index.numel() > 0:
            edge_parts.append(snapshot.edge_index + offset)
            if use_edge_weight and hasattr(snapshot, "edge_weight"):
                edge_weight_parts.append(snapshot.edge_weight)
        offset += int(snapshot.num_nodes)

    edge_index = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long)
    )
    data = Data(
        x=torch.cat(x_parts, dim=0),
        edge_index=edge_index,
        y=torch.cat(y_parts, dim=0),
        base_pred=torch.cat(base_parts, dim=0),
        true_value=torch.cat(true_parts, dim=0),
        location_id=torch.cat(location_parts, dim=0),
        prior_score=torch.cat(prior_parts, dim=0),
        stability_score=torch.cat(stability_parts, dim=0),
        history_consistency_score=torch.cat(history_parts, dim=0),
        full_confidence=torch.cat(confidence_parts, dim=0),
    )
    if use_edge_weight:
        data.edge_weight = (
            torch.cat(edge_weight_parts, dim=0)
            if edge_weight_parts
            else torch.empty((0,), dtype=torch.float32)
        )
    data.confidence_feature_index = 1
    return data
