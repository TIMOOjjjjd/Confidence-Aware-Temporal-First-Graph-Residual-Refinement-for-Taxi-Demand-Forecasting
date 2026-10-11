"""Measured CPU residual-fit worker scaling, NOT complete TCN pipeline speedup."""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch_geometric.data import Data

from experiment_2_from_tcn import load_no_graph_tcn_baseline, build_step_df_from_tcn
from experiment_3_confidence_weighted_gnn_ablation import (
    GNNTrainingConfig, build_graph_context, build_zone_hourly_counts,
    compute_prior_scores_from_zone_hourly_counts, make_location_split_sets,
    masks_from_location_splits, train_residual_gnn_with_weighted_loss,
)
from prepare_revision_data import load_cached_trip_data
from revision_protocol import sha256_file
from tcn_feature_utils import (
    build_gnn_features_with_history, build_historical_gnn_training_data_with_history,
    select_residual_training_history,
)


BASE = Path(__file__).resolve().parent
_PAYLOAD = None


def initialize_worker(path, threads):
    global _PAYLOAD
    torch.set_num_threads(threads)
    _PAYLOAD = torch.load(path, map_location="cpu", weights_only=False)


def fit_worker(seed):
    payload = _PAYLOAD
    train, inference = Data(**payload["train"]), Data(**payload["inference"])
    split = make_location_split_sets(payload["zones"], 42, .6, .2)
    masks = masks_from_location_splits(train, split, require_train_val=True, require_test=False)
    started = time.perf_counter()
    result = train_residual_gnn_with_weighted_loss(
        train, masks, inference, "none", torch.device("cpu"),
        GNNTrainingConfig(**payload["config"]), seed,
    )
    return {"seed": seed, "fit_seconds": time.perf_counter()-started,
            "best_epoch": result.best_epoch,
            "predictions": len(result.refined_pred),
            "rss_bytes": psutil.Process().memory_info().rss}


def measure_workers(path, workers, repeats, threads):
    stop = threading.Event()
    peak = [0]
    parent = psutil.Process()

    def sample():
        while not stop.is_set():
            processes = [parent] + parent.children(recursive=True)
            total = 0
            for process in processes:
                try:
                    total += process.memory_info().rss
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], total)
            stop.wait(.05)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                 initargs=(str(path), threads)) as executor:
            results = list(executor.map(fit_worker, range(42, 42+repeats)))
    finally:
        stop.set()
        monitor.join()
    elapsed = time.perf_counter()-started
    return {"workers": workers, "fits": repeats, "wall_seconds_including_startup": elapsed,
            "fits_per_second": repeats/elapsed, "peak_aggregate_rss_bytes": peak[0],
            "mean_fit_seconds": np.mean([row["fit_seconds"] for row in results])}, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tcn-detail", type=Path, required=True)
    parser.add_argument("--cohort-file", type=Path, required=True)
    parser.add_argument("--edge-csv", type=Path, default=BASE / "edge_weight_matrix_geo.csv")
    parser.add_argument("--target-hour", type=pd.Timestamp,
                        default=pd.Timestamp("2021-03-01 23:00"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    args = parser.parse_args()
    if min(*args.workers, args.repeats, args.epochs, args.hidden_dim, args.threads_per_worker) < 1:
        parser.error("Positive worker counts and training settings required.")
    if args.repeats < max(args.workers):
        parser.error("At least one fit per worker required.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Runtime output must be new.")
    torch.set_num_threads(args.threads_per_worker)
    started = time.perf_counter()
    cache = load_no_graph_tcn_baseline(args.tcn_detail, "base_pred")
    zones = json.loads(args.cohort_file.read_text(encoding="utf-8"))
    counts = build_zone_hourly_counts(load_cached_trip_data(args.data))
    graph = build_graph_context(args.edge_csv, pd.read_csv(BASE / "taxi-zone-lookup.csv"))
    window_start = args.target_hour.floor("D")
    frames = []
    for hour in pd.date_range(window_start, args.target_hour, freq="h"):
        frames.append(build_step_df_from_tcn(
            cache, hour, zones, counts, compute_prior_scores_from_zone_hourly_counts(counts, hour)
        ))
    history, _ = select_residual_training_history(
        frames[:-1], args.target_hour, {6, 7, 8, 9, 16, 17, 18}, "peak_non_peak"
    )
    train = build_historical_gnn_training_data_with_history(history, graph, True)
    inference = build_gnn_features_with_history(frames[-1], graph, True)
    if train is None:
        raise ValueError("No pre-target residual history for runtime measurement.")
    preparation_seconds = time.perf_counter()-started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "frozen_residual_inputs.pt"
    torch.save({"train": train.to_dict(), "inference": inference.to_dict(), "zones": zones,
                "config": {"epochs": args.epochs, "hidden_dim": args.hidden_dim,
                           "patience": args.epochs}}, path)
    records, fit_records = [], []
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(args.threads_per_worker)
    for workers in args.workers:
        result, fits = measure_workers(path, workers, args.repeats, args.threads_per_worker)
        records.append(result)
        fit_records.extend({"workers": workers, **row} for row in fits)
        print(result, flush=True)
    summary = pd.DataFrame(records)
    one_worker = summary.loc[summary.workers.eq(1), "wall_seconds_including_startup"]
    summary["speedup_vs_one_worker"] = (one_worker.iloc[0]/summary.wall_seconds_including_startup
                                        if len(one_worker) else np.nan)
    summary.to_csv(args.output_dir / "runtime_summary.csv", index=False)
    pd.DataFrame(fit_records).to_csv(args.output_dir / "individual_fits.csv", index=False)
    (args.output_dir / "runtime_manifest.json").write_text(json.dumps({
        "arguments": vars(args), "cache_sha256": sha256_file(args.tcn_detail),
        "preparation_seconds": preparation_seconds,
        "scope": "independent CPU residual fits on one identical real-data snapshot",
        "excludes": ["TCN training", "TCN inference", "rolling OD construction"],
        "memory": "sampled sum of parent/child RSS; shared pages may be counted more than once",
    }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
