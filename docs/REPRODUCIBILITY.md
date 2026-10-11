# Reproducibility workflow

## Protocol shared by the paper experiments

- Evaluation windows begin at 00:00 on 1 March, 29 March, 26 April, 24 May, 21 June, 5 July, 19 July, 16 August, 13 September, 11 October, 8 November, and 6 December 2021.
- Each window contains 24 one-hour-ahead targets.
- Excluded identifiers are 46, 103, 104, 105, 264, and 265.
- Revised residual experiments use a fixed 60/20/20 location split and seeds 11, 23, 42, 67, and 89 where stated below.
- Main Ex2, Ex3, and Ex5 results use residual multiplier `1.0`.
- Legacy Ex4 and Ex6 results use peak multiplier `0.3`.
- OD60 admits only trips completed before the target hour.

All commands below are run from the repository root. Output directories must be new because the formal scripts refuse to mix or overwrite experiment runs.

## 1. Temporal backbone comparison Ex1

```bash
python code/experiment_1_temporal_backbones.py \
  --data code/data.parquet \
  --results-dir outputs/ex1 \
  --backends sarima dlinear gru lstm patchtst transformer tcn
```

This is the most expensive stage because it trains multiple temporal models over all evaluated zones.

## 2. Recreate the shared TCN prediction cache

The released TCN checkpoints allow the temporal predictions to be generated without retraining:

```bash
python code/experiment_5_graph_structure_comparison.py \
  --data code/data.parquet \
  --lookup code/taxi-zone-lookup.csv \
  --geo-edge-csv code/edge_weight_matrix_geo.csv \
  --checkpoint-dir code/checkpoints_tcn_shared_v1 \
  --results-dir outputs/tcn_reference \
  --graph-types no_graph \
  --no-plot
```

Freeze the no-graph rows into the cache format consumed by Ex2--Ex6:

```bash
python code/freeze_tcn_cache.py \
  --source outputs/tcn_reference/experiment_5_graph_structure_comparison_detailed.csv \
  --output-dir artifacts/tcn_cache
```

The paper revision used a 74,592-row cache with 259 locations and 288 target hours. Its recorded SHA-256 was:

```text
794a3824f53a8b14faf30a83ce3df5798cbd41593a87127ff5de4dda3684bfb3
```

Verify the regenerated cache before treating subsequent metrics as an exact replay.

## 3. Prepare reusable residual-stage data

```bash
python code/prepare_revision_data.py \
  --data code/data.parquet \
  --output-dir artifacts/prepared_data
```

## 4. Ex2 feature ablation seed 42

```bash
python code/run_revision_suite.py \
  --suite features \
  --seeds 42 \
  --data artifacts/prepared_data \
  --tcn-detail artifacts/tcn_cache/baseline.csv \
  --edge-csv code/edge_weight_matrix_geo.csv \
  --output-dir outputs/ex2 \
  --execute
```

## 5. Ex3 confidence feature by loss-weighting ablation

```bash
python code/run_revision_suite.py \
  --suite confidence \
  --seeds 11 23 42 67 89 \
  --data artifacts/prepared_data \
  --tcn-detail artifacts/tcn_cache/baseline.csv \
  --edge-csv code/edge_weight_matrix_geo.csv \
  --output-dir outputs/ex3 \
  --execute
```

## 6. Ex4 legacy confidence-weight search seed 42

```bash
python code/experiment_4_from_tcn.py \
  --data artifacts/prepared_data \
  --tcn-detail artifacts/tcn_cache/baseline.csv \
  --edge-csv code/edge_weight_matrix_geo.csv \
  --results-dir outputs/ex4 \
  --seed 42 \
  --peak-shrinkage-alpha 0.3 \
  --modes fixed grid_search random_search learned_softmax \
  --no-plot
```

The paper labels Ex4 as legacy sensitivity evidence. Its reported absolute errors use peak residual multiplier `0.3`; the primary revised controls use `1.0`.

## 7. Ex5 residual and graph controls

```bash
python code/run_revision_suite.py \
  --suite graph \
  --seeds 11 23 42 67 89 \
  --data artifacts/prepared_data \
  --tcn-detail artifacts/tcn_cache/baseline.csv \
  --edge-csv code/edge_weight_matrix_geo.csv \
  --output-dir outputs/ex5 \
  --execute
```

The graph suite evaluates temporal-only, capacity-matched residual MLP, edge-free GraphSAGE, geographic GraphSAGE, and completed-trip OD60 GraphSAGE controls.

## 8. Ex6 legacy reliability analysis

```bash
python code/experiment_6_from_tcn.py \
  --data artifacts/prepared_data \
  --tcn-detail artifacts/tcn_cache/baseline.csv \
  --edge-csv code/edge_weight_matrix_geo.csv \
  --results-dir outputs/ex6 \
  --seed 42 \
  --peak-shrinkage-alpha 0.3 \
  --mode random_search \
  --no-plot
```

## 9. Matched analysis

Use the detailed Ex3 or Ex5 prediction CSVs with:

```bash
python code/analyze_revision_predictions.py --help
python code/analyze_confidence_controls.py --help
python code/analyze_peak_tcn_calibration.py --help
```

For graph attribution, compare the same Ex5 prediction rows against both the residual MLP and edge-free GraphSAGE controls. The day-block intervals resample the twelve selected days and keep the five observed residual seeds fixed.

## 10. Runtime experiment Ex8

```bash
python code/experiment_8_runtime_scaling.py --help
```

Run this only when no other training workload is competing for CPU or memory. It measures independent residual fits and excludes temporal computation, OD construction, and full orchestration.

## Interpretation boundary

Successful execution reproduces the implementation and protocol. A smoke test, a subset of zones, fewer epochs, or fewer windows does not reproduce the formal paper evidence. The code does not establish end-to-end spatiotemporal-GNN superiority, cross-city generalization, or calibrated confidence probabilities.
