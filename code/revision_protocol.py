"""Shared, cache-only protocol for reviewer revision experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd


def add_revision_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=None,
                        help="Model initialization seed; defaults to --seed.")
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--residual-model", choices=["graphsage", "mlp", "node_sage"],
                        default="graphsage")
    parser.add_argument("--exclude-confidence-feature", action="store_true")
    parser.add_argument("--expected-cache-sha256", default=None)
    parser.add_argument("--cohort-file", type=Path, default=None,
                        help="Optional frozen JSON list of location IDs.")
    parser.add_argument("--torch-threads", type=int, default=2)


def resolve_revision_args(args: argparse.Namespace) -> None:
    if args.model_seed is None:
        args.model_seed = args.seed
    args.seed = args.model_seed
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    import torch

    torch.set_num_threads(args.torch_threads)
    if args.gnn_epochs < 1 or args.gnn_hidden_dim < 1 or args.gnn_patience < 1:
        raise ValueError("GNN epochs, width and patience must be positive.")
    if not 0 < args.train_ratio < 1 or not 0 <= args.val_ratio < 1:
        raise ValueError("Invalid location split ratios.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("The location split must leave test nodes.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def start_revision_run(args: argparse.Namespace, entry_file: str) -> None:
    """Reserve a NEW output directory before any graph/result writes."""
    resolve_revision_args(args)
    output = Path(args.results_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    for key in ["od_graph_dir", "random_graph_dir"]:
        path = getattr(args, key, None)
        if path is not None and Path(path).exists() and any(Path(path).iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing graph cache: {path}")
    cache = Path(args.tcn_detail).resolve()
    cache_hash = sha256_file(cache)
    if args.expected_cache_sha256 and cache_hash != args.expected_cache_sha256.lower():
        raise ValueError("TCN cache SHA256 does not match the frozen cache.")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(entry_file).resolve().parent
    sources = {
        path.name: sha256_file(path) for path in sorted(source_dir.glob("*.py"))
    }
    from gnn_model import residual_capacity

    input_dim = 5 - int(args.exclude_confidence_feature)
    manifest = {
        "protocol_version": "cached_tcn_revision_v1",
        "tcn_training": "disabled; all forecasts read from cache",
        "uncertainty_scope": "conditional on fixed temporal forecasts",
        "entry_point": str(Path(entry_file).resolve()),
        "cache": {"path": str(cache), "sha256": cache_hash,
                  "prediction_column": args.prediction_column},
        "arguments": {key: value for key, value in vars(args).items()
                      if not key.startswith("_")},
        "python": platform.python_version(),
        "source_sha256": sources,
        "history_rich_model_capacity": {
            kind: residual_capacity(input_dim, args.gnn_hidden_dim, kind)
            for kind in ["graphsage", "mlp", "node_sage"]
        },
        "prepared_data_manifest_sha256": sha256_file(Path(args.data)/"data_manifest.json")
        if Path(args.data).is_dir() else None,
        "analysis_prediction_sha256": sha256_file(args.detailed_csv)
        if getattr(args, "detailed_csv", None) is not None else None,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    args._revision_started = time.perf_counter()
    args._cache_sha256 = cache_hash


def freeze_cohort(args, location_ids, cache, window_starts):
    """Keep one structural node cohort and require complete finite cached labels."""
    locations = sorted(set(int(value) for value in location_ids))
    if args.cohort_file is not None:
        requested = sorted(set(json.loads(args.cohort_file.read_text(encoding="utf-8"))))
        missing = set(requested).difference(locations)
        if missing:
            raise ValueError(f"Frozen cohort incompatible with selected graphs: {missing}")
        locations = requested
    hours = [pd.Timestamp(start) + pd.Timedelta(hours=step)
             for start in window_starts for step in range(args.rolling_steps)]
    if len(hours) != len(set(hours)):
        raise ValueError("Overlapping evaluation windows would duplicate predictions.")
    expected = pd.MultiIndex.from_product(
        [hours, locations], names=["target_hour", "PULocationID"]
    )
    aligned = cache.reindex(expected)
    valid = np.isfinite(aligned[["base_pred", "true_value"]].to_numpy(dtype=float)).all(1)
    if not valid.all():
        missing = expected[~valid].tolist()[:10]
        raise ValueError(f"TCN cache is incomplete/nonfinite for the cohort: {missing}")
    (Path(args.results_dir) / "cohort.json").write_text(
        json.dumps(locations), encoding="utf-8"
    )
    return locations


def annotate_predictions(frame, args, trained=True):
    frame = frame.copy()
    frame["model_seed"] = args.model_seed
    frame["split_seed"] = args.split_seed
    frame["search_seed"] = args.search_seed
    frame["residual_model"] = args.residual_model
    frame["confidence_feature_used"] = not args.exclude_confidence_feature
    if "best_epoch" in frame:
        frame["fallback"] = frame["best_epoch"].fillna(0).eq(0)
    else:
        frame["fallback"] = not trained
    if "model_id" in frame:
        frame["confidence_feature_used"] &= frame.model_id.isin(["G1", "G3"])
        frame.loc[frame.model_id.eq("T4"), "residual_model"] = "temporal"
    if "graph_type" in frame:
        frame.loc[frame.graph_type.eq("no_graph"), "confidence_feature_used"] = False
        frame.loc[frame.graph_type.eq("no_graph"), "residual_model"] = "temporal"
    return frame


def finish_revision_run(args) -> None:
    if sha256_file(args.tcn_detail) != args._cache_sha256:
        raise ValueError("TCN cache changed during the run; results are not frozen-cache valid.")
    elapsed = time.perf_counter() - args._revision_started
    (Path(args.results_dir) / "run_status.json").write_text(
        json.dumps({"status": "completed", "elapsed_seconds": elapsed,
                    "scope": "cached temporal forecasts; includes data/graph/residual work"},
                   indent=2), encoding="utf-8"
    )
