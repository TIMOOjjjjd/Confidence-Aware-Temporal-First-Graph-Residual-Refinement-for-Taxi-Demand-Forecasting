"""Stream raw trips into reusable hourly demand and hourly-available OD events.

No temporal model is loaded or trained. Availability bins preserve STRICT
drop-off < target for hour-aligned forecasts, including exact-hour drop-offs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def aggregate_batch(frame):
    frame = frame.copy()
    pickup = pd.to_datetime(frame["pickup_datetime"])
    dropoff = pd.to_datetime(frame["dropoff_datetime"])
    frame["pickup_datetime"] = pickup.dt.floor("h")
    frame["available_from"] = dropoff.dt.floor("h") + pd.Timedelta(hours=1)
    frame.loc[dropoff < pickup, "available_from"] = pd.NaT
    counts = frame.groupby(["PULocationID", "pickup_datetime"]).size()
    events = frame.groupby(
        ["pickup_datetime", "available_from", "PULocationID", "DOLocationID"],
        dropna=False,
    ).size().rename("trip_count").reset_index()
    return counts.rename("trip_count"), events


def prepare(source, output, batch_size=1_000_000):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    events_dir = output / "od_events"
    events_dir.mkdir()
    parquet = pq.ParquetFile(source)
    counts = []
    rows = 0
    columns = ["pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID"]
    for index, batch in enumerate(parquet.iter_batches(batch_size=batch_size, columns=columns)):
        hourly, events = aggregate_batch(batch.to_pandas())
        counts.append(hourly)
        events.to_parquet(events_dir / f"part-{index:05d}.parquet", index=False)
        rows += batch.num_rows
        if index % 16 == 15:
            counts = [pd.concat(counts).groupby(level=[0, 1]).sum()]
        print(f"Prepared {rows:,}/{parquet.metadata.num_rows:,} trips", flush=True)
    hourly = pd.concat(counts).groupby(level=[0, 1]).sum().reset_index()
    hourly.to_parquet(output / "hourly_counts.parquet", index=False)
    stat = source.stat()
    (output / "data_manifest.json").write_text(json.dumps({
        "format": "hourly_demand_available_od_v1", "source": str(source.resolve()),
        "source_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "rows": rows, "hourly_rows": len(hourly),
        "availability": "floor(dropoff)+1h; invalid completion times unavailable",
    }, indent=2), encoding="utf-8")


def load_cached_trip_data(path, include_dropoff=False):
    path = Path(path)
    if path.is_dir():
        manifest = path / "data_manifest.json"
        if not manifest.exists():
            raise ValueError("Prepared trip data incomplete: no data manifest.")
        frame = pd.read_parquet(path / "hourly_counts.parquet")
        if include_dropoff:
            frame.attrs["od_events_path"] = str((path / "od_events").resolve())
    else:
        columns = ["pickup_datetime", "PULocationID"]
        if include_dropoff:
            columns += ["DOLocationID", "dropoff_datetime"]
        frame = pd.read_parquet(path, columns=columns)
        if include_dropoff:
            frame["dropoff_datetime"] = pd.to_datetime(frame["dropoff_datetime"])
    frame["pickup_datetime"] = pd.to_datetime(frame["pickup_datetime"])
    frame["datetime"] = frame["pickup_datetime"].dt.floor("h")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    prepare(args.data, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
