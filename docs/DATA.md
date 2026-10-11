# Data preparation

## Download

`data.parquet` is intentionally excluded because of its size.

download data.parquet from: [Google Drive](https://drive.google.com/file/d/1qpKjFBmlLIpIlfIITGhIRqbT8YEfjkec/view?usp=drive_link)

Save it as `code/data.parquet`.

## Expected schema

The formal preprocessing reads:

- `pickup_datetime`
- `dropoff_datetime`
- `PULocationID`
- `DOLocationID`

The file contains trip-level records. Do not commit it or the generated `od_events` partitions.

## Prepared residual data

Generate hourly pickup counts and bounded OD-event partitions:

```bash
python code/prepare_revision_data.py \
  --data code/data.parquet \
  --output-dir artifacts/prepared_data
```

For a target hour `t`, the completed-trip OD graph uses pickups in the historical lookback window only when the trip has a valid drop-off strictly before `t`. The preprocessing represents this as `available_from = floor(dropoff_datetime) + 1 hour`. Trips with a drop-off before pickup are not made available to the completed-trip graph.

Pickup and drop-off timestamps are parsed without timezone localization or UTC conversion. All experiment boundaries therefore use the recorded New York source clock.

## Static resources

- `code/taxi-zone-lookup.csv` maps taxi-zone identifiers to names.
- `code/taxi_zones.geojson` is the archived taxi-zone geometry resource.
- `code/edge_weight_matrix_geo.csv` is the fixed 258-by-258 geographic matrix evaluated by the paper.
- `code/edge_weight_matrix_od.csv` is retained for legacy/static OD experiment compatibility.

The generator for `edge_weight_matrix_geo.csv` was not preserved. The file is supplied as an inspectable fixed experimental input; this repository does not claim that the topology can be regenerated from the included GeoJSON.
