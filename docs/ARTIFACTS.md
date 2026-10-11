# Released artifacts

## Frozen TCN checkpoints

`code/checkpoints_tcn_shared_v1/` contains 259 model files, 259 fitted scalers, and 259 metadata files. The directory totals approximately 88 MB.

Each metadata file records `2021-02-28 23:00:00` as the last training-context hour. These checkpoints support regeneration of the temporal predictions used to create the frozen residual-stage cache. Loading Python pickle files should only be done from a trusted checkout of this repository.

The fitted scaler pickles were created with scikit-learn 1.7.2, which is pinned in `requirements.txt` to avoid cross-version deserialization warnings.

The paper's repeated Ex3 and Ex5 seeds alter residual-model initialization while sharing one temporal cache. They do not retrain these TCN checkpoints.

## Excluded artifacts

The following are intentionally not released in the Git repository:

- `data.parquet`
- prepared OD-event partitions
- the frozen TCN prediction CSV
- detailed per-zone revision prediction tables
- large intermediate result directories
- submission PDFs, Licence-to-Publish forms, and correspondence

The frozen prediction cache can be regenerated from `data.parquet` and the released checkpoints using the workflow in `REPRODUCIBILITY.md`.
