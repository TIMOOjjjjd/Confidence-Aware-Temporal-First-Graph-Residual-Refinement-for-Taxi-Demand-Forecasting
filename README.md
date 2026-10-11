# Confidence-Aware Temporal-First Graph Residual Refinement for Taxi Demand Forecasting

Code accompanying the ICA3PP 2026 paper **Confidence-Aware Temporal-First Graph Residual Refinement for Taxi Demand Forecasting**.

## Paper authors

- Tianqi Wang ([tim.wang@warwick.ac.uk](mailto:tim.wang@warwick.ac.uk))
- Ligang He ([ligang.he@warwick.ac.uk](mailto:ligang.he@warwick.ac.uk), corresponding author)

The paper is available as [`main.pdf`](main.pdf). Repository citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Scope

The implementation follows the paper's temporal-first protocol:

1. Train or load per-zone temporal forecasting models.
2. Freeze the TCN predictions used by the residual experiments.
3. Fit residual MLP, edge-free GraphSAGE, geographic GraphSAGE, or rolling-OD GraphSAGE models using only historical labels.
4. Optionally use the bounded confidence score as a node feature or residual-loss weight.

The repository covers the paper's Ex1--Ex6 experiments and the residual-stage runtime experiment Ex8. The unfinished Graph WaveNet/AGCRN comparison is intentionally excluded and is not part of the paper evidence.

## Data

`data.parquet` is too large for this repository and is not committed.

download data.parquet from: [Google Drive](https://drive.google.com/file/d/1qpKjFBmlLIpIlfIITGhIRqbT8YEfjkec/view?usp=drive_link)

Place the downloaded file at:

```text
code/data.parquet
```

The scripts expect these columns:

```text
pickup_datetime
dropoff_datetime
PULocationID
DOLocationID
```

The paper uses 2021 NYC TLC trip records. Timestamps are treated as timezone-naive source-clock values. See [`docs/DATA.md`](docs/DATA.md) for preprocessing and causality details.

## Environment

The release was validated with Python 3.11.7. Create an isolated environment and install the recorded dependencies:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validation

Run the deterministic CPU tests:

```bash
python -m pytest -q
```

Run a minimal TCN-cache smoke test after placing `data.parquet` under `code/`:

```bash
python code/experiment_5_graph_structure_comparison.py \
  --data code/data.parquet \
  --lookup code/taxi-zone-lookup.csv \
  --geo-edge-csv code/edge_weight_matrix_geo.csv \
  --checkpoint-dir code/checkpoints_tcn_shared_v1 \
  --results-dir outputs/smoke_tcn \
  --window-starts "2021-03-01 00:00" \
  --rolling-steps 1 \
  --max-zones 3 \
  --graph-types no_graph \
  --no-plot
```

This smoke test checks loading, preprocessing, checkpoint inference, and CSV generation. It does not reproduce the formal paper metrics.

## Reproducing the paper implementation

The exact workflow, frozen seeds, window schedule, cache generation, and Ex2--Ex6 commands are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

The full experiments are CPU-intensive. Ex3 and Ex5 use residual-model seeds `11 23 42 67 89`, 12 non-contiguous 24-hour windows, 51 fixed test locations, hidden width 256, at most 300 epochs, and early-stopping patience 40. These repeated residual seeds condition on one frozen TCN cache; they are not five independent TCN retrainings.

## Evidence boundaries

- Residual refinement produces the clearest performance gain.
- Geographic GraphSAGE has the lowest mean MAE, but its paired day-block intervals against matched non-graph controls include zero.
- Confidence features and weighting provide a smaller empirical improvement and are not calibrated probabilities.
- Ex8 measures process-level parallelism for independent residual fits, not end-to-end pipeline scalability.
- Cross-city evaluation, continuous-year evaluation, temporal-backbone retraining uncertainty, and formal end-to-end Graph WaveNet/AGCRN comparisons are not included.

## Repository contents

- `main.pdf`: paper manuscript, including author affiliations and contact emails.
- `code/`: paper experiment implementations and frozen per-zone TCN checkpoints.
- `tests/`: deterministic regression tests that do not train the temporal model.
- `docs/DATA.md`: dataset placement, schema, preprocessing, and redistribution boundary.
- `docs/REPRODUCIBILITY.md`: commands and protocol for Ex1--Ex6 and Ex8.
- `docs/ARTIFACTS.md`: supplied checkpoint and topology provenance.

Generated results, prepared trip-level data, caches, virtual environments, and submission documents are intentionally ignored.

## License

No software licence has been selected yet. The source is publicly inspectable, but reuse permissions should not be inferred until the authors add an explicit licence after confirming ownership requirements.
