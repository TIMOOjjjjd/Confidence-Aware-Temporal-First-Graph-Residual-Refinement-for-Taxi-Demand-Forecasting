"""Budgeted multi-seed jobs using ONE frozen TCN cache and prepared trip data."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiment_2_from_tcn import load_no_graph_tcn_baseline
from experiment_3_confidence_weighted_gnn_ablation import build_graph_context, EXCLUDED_ZONES
from revision_protocol import sha256_file


BASE = Path(__file__).resolve().parent


def build_jobs(args, cache_hash, cohort_path):
    jobs = []
    for seed in args.seeds:
        common = ["--data", str(args.data.resolve()), "--tcn-detail", str(args.tcn_detail.resolve()),
                  "--expected-cache-sha256", cache_hash, "--model-seed", str(seed),
                  "--split-seed", str(args.split_seed), "--search-seed", str(args.search_seed),
                  "--cohort-file", str(cohort_path.resolve()), "--gnn-epochs", str(args.epochs),
                  "--gnn-patience", str(args.patience), "--gnn-hidden-dim", str(args.hidden_dim),
                  "--rolling-steps", str(args.rolling_steps), "--peak-shrinkage-alpha", "1",
                  "--torch-threads", str(args.torch_threads), "--no-plot"]
        if args.window_starts:
            common += ["--window-starts", *args.window_starts]
        if args.max_zones:
            common += ["--max-zones", str(args.max_zones)]
        experiments = []
        if args.suite in ["all", "graph"]:
            experiments.append(("graph", "experiment_5_from_tcn.py", [
                "--graph-types", "no_graph", "residual_mlp", "node_sage", "geo", "od",
                "--od-lookback-days-list", "60", "--geo-edge-csv", str(args.edge_csv.resolve()),
            ]))
        if args.suite in ["all", "confidence"]:
            for use_feature in [False, True]:
                extra = ["--modes", "none", "full", "--edge-csv", str(args.edge_csv.resolve())]
                if not use_feature:
                    extra += ["--exclude-confidence-feature"]
                experiments.append((f"confidence_feature_{int(use_feature)}",
                                    "experiment_3_from_tcn.py", extra))
        if args.suite in ["all", "features"]:
            experiments.append(("features", "experiment_2_from_tcn.py", [
                "--confidence-weight-mode", "uniform", "--edge-csv", str(args.edge_csv.resolve()),
            ]))
        if args.suite in ["all", "weights"]:
            experiments.append(("weights", "experiment_4_from_tcn.py", [
                "--modes", "fixed", "grid_search", "random_search", "learned_softmax",
                "--random-candidates", str(args.random_candidates),
                "--fixed-confidence-weights", "0.3333333333", "0.3333333333", "0.3333333333",
                "--edge-csv", str(args.edge_csv.resolve()),
            ]))
        for label, script, extra in experiments:
            directory = args.output_dir / f"{label}_seed_{seed}"
            jobs.append({"label": label, "seed": seed, "output": str(directory.resolve()),
                         "command": [sys.executable, str(BASE / script), *common, *extra,
                                     "--results-dir", str(directory.resolve())]})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE / "results/revision_data_20260917")
    parser.add_argument("--tcn-detail", type=Path,
                        default=BASE / "results/revision_tcn_cache_20260917/baseline.csv")
    parser.add_argument("--edge-csv", type=Path, default=BASE / "edge_weight_matrix_geo.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=["graph", "confidence", "features", "weights", "all"],
                        default="graph")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 42, 67, 89])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--random-candidates", type=int, default=24)
    parser.add_argument("--rolling-steps", type=int, default=24)
    parser.add_argument("--window-starts", nargs="+", default=None)
    parser.add_argument("--max-zones", type=int, default=None)
    parser.add_argument("--execute", action="store_true",
                        help="Without this flag only a frozen job plan is produced.")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique.")
    if not args.data.is_dir() or not (args.data / "data_manifest.json").exists():
        parser.error("Run prepare_revision_data.py first; shared prepared data required.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Suite output directory must be new.")
    cache = load_no_graph_tcn_baseline(args.tcn_detail, "base_pred")
    graph = build_graph_context(args.edge_csv, pd.read_csv(BASE / "taxi-zone-lookup.csv"))
    zones = sorted(set(graph.zone_to_location.values()).intersection(
        cache.index.get_level_values("PULocationID")) - set(EXCLUDED_ZONES))
    if args.max_zones:
        zones = zones[:args.max_zones]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = args.output_dir / "cohort.json"
    cohort_path.write_text(json.dumps(zones), encoding="utf-8")
    cache_hash = sha256_file(args.tcn_detail)
    jobs = build_jobs(args, cache_hash, cohort_path)
    sources = {path.name: sha256_file(path) for path in BASE.glob("*.py")}
    (args.output_dir / "suite_plan.json").write_text(json.dumps({
        "tcn_training": "disabled", "cache_sha256": cache_hash,
        "source_sha256": sources,
        "arguments": vars(args), "jobs": jobs,
    }, indent=2, default=str), encoding="utf-8")
    print(f"Frozen {len(jobs)} jobs; {len(zones)} locations; TCN training disabled.")
    if not args.execute:
        return
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.torch_threads)
    environment["MKL_NUM_THREADS"] = str(args.torch_threads)
    for index, job in enumerate(jobs):
        if any(sha256_file(BASE/name) != digest for name, digest in sources.items()):
            raise RuntimeError("Experiment code changed during the suite; refusing mixed versions.")
        print(f"Running {index+1}/{len(jobs)}: {job['label']} seed={job['seed']}", flush=True)
        log = args.output_dir / f"{job['label']}_seed_{job['seed']}.log"
        with log.open("x", encoding="utf-8") as handle:
            subprocess.run(job["command"], check=True, stdout=handle,
                           stderr=subprocess.STDOUT, env=environment)
    (args.output_dir / "suite_status.json").write_text(
        json.dumps({"status": "completed", "jobs": len(jobs)}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
