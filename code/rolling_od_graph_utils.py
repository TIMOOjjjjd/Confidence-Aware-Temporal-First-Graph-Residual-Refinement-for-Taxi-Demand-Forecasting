"""Shared rolling OD graph construction used by graph experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Sequence, Set, Tuple

import pandas as pd

from build_od_graph import (
    _build_matrix,
    _load_location_lookup,
    _load_zone_names,
    _retain_top_k,
)


DEFAULT_OD_LOOKBACK_DAYS = 30


def add_rolling_od_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--od-availability", choices=["completed", "pickup"],
                        default="completed",
                        help="Completed trips only; pickup is a legacy sensitivity control.")
    parser.add_argument(
        "--od-lookback-days",
        type=int,
        default=DEFAULT_OD_LOOKBACK_DAYS,
        help=(
            "Build OD graphs from trips in [target_hour - N days, target_hour). "
            "Use 0 to load the static --edge-csv matrix."
        ),
    )
    parser.add_argument(
        "--od-graph-dir",
        type=Path,
        default=None,
        help="Directory for generated rolling OD graph CSVs. Defaults to results/od_graphs.",
    )
    parser.add_argument(
        "--od-top-k",
        type=int,
        default=10,
        help="Keep the top K destination zones per origin when generating rolling OD graphs.",
    )
    parser.add_argument(
        "--od-min-flow",
        type=int,
        default=1,
        help="Drop rolling OD edges with fewer than this many trips before top-k filtering.",
    )
    parser.add_argument(
        "--od-weight-mode",
        choices=["row_share", "retained_share", "count", "binary", "log_count"],
        default="row_share",
        help="How to write retained rolling OD edge weights.",
    )
    parser.add_argument(
        "--od-symmetrize",
        choices=["none", "max", "sum", "mean"],
        default="none",
        help="Optionally convert each rolling OD graph to an undirected matrix.",
    )
    parser.add_argument(
        "--od-include-self",
        action="store_true",
        help="Keep trips where pickup and dropoff map to the same zone in rolling OD graphs.",
    )


def validate_rolling_od_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.od_lookback_days < 0:
        parser.error("--od-lookback-days must be >= 0.")
    if args.od_top_k < 0:
        parser.error("--od-top-k must be >= 0.")
    if args.od_min_flow < 1:
        parser.error("--od-min-flow must be >= 1.")


def uses_rolling_od_graph(args: argparse.Namespace) -> bool:
    return getattr(args, "graph_type", None) == "od" and int(args.od_lookback_days) > 0


def od_graph_dir(args: argparse.Namespace) -> Path:
    return args.od_graph_dir if args.od_graph_dir is not None else args.results_dir / "od_graphs"


def od_graph_path(args: argparse.Namespace, target_hour: pd.Timestamp) -> Path:
    timestamp = pd.Timestamp(target_hour).strftime("%Y%m%d_%H%M%S")
    return od_graph_dir(args) / (
        f"edge_weight_matrix_od_last_{args.od_lookback_days}d_until_{timestamp}.csv"
    )


def load_od_zone_names(args: argparse.Namespace) -> list[str]:
    template_path = args.edge_csv if args.edge_csv.exists() else None
    return _load_zone_names(args.lookup, template_path=template_path)


def load_od_location_lookup(args: argparse.Namespace) -> Dict[int, str]:
    return _load_location_lookup(args.lookup)


def aggregate_od_flows_from_frame(
    frame: pd.DataFrame,
    location_to_zone: Dict[int, str],
    excluded_locations: Set[int],
    include_self: bool,
) -> Tuple[Counter[Tuple[str, str]], Counter[str], int]:
    flows: Counter[Tuple[str, str]] = Counter()
    origin_totals: Counter[str] = Counter()

    od_frame = frame[["PULocationID", "DOLocationID"]].dropna().copy()
    if od_frame.empty:
        return flows, origin_totals, 0

    od_frame["PULocationID"] = od_frame["PULocationID"].astype(int)
    od_frame["DOLocationID"] = od_frame["DOLocationID"].astype(int)

    if excluded_locations:
        od_frame = od_frame[
            ~od_frame["PULocationID"].isin(excluded_locations)
            & ~od_frame["DOLocationID"].isin(excluded_locations)
        ]
    if od_frame.empty:
        return flows, origin_totals, 0

    if "trip_count" in frame:
        od_frame["trip_count"] = frame.loc[od_frame.index, "trip_count"]
        grouped = od_frame.groupby(["PULocationID", "DOLocationID"], sort=False)["trip_count"].sum()
    else:
        grouped = od_frame.groupby(["PULocationID", "DOLocationID"], sort=False).size()
    for (origin_id, dest_id), count in grouped.items():
        origin, dest = location_to_zone.get(origin_id), location_to_zone.get(dest_id)
        if origin is None or dest is None or (not include_self and origin == dest):
            continue
        count_int = int(count)
        flows[(str(origin), str(dest))] += count_int
        origin_totals[str(origin)] += count_int

    return flows, origin_totals, int(sum(flows.values()))


def aggregate_available_od(args, df, target_hour, lookback_days, location_to_zone):
    """Bounded-memory aggregation; increment exact rolling flows between targets."""
    import pyarrow.dataset as ds

    end = pd.Timestamp(target_hour)
    start = end - pd.Timedelta(days=lookback_days)
    availability = getattr(args, "od_availability", "completed")
    if "od_events_path" in df.attrs and end != end.floor("h"):
        raise ValueError("Prepared OD events support hour-aligned forecasts only.")
    states = getattr(args, "_od_flow_states", {})
    key = (lookback_days, availability)

    def scan(pickup_start, pickup_end, completed_before, completed_after=None):
        if "od_events_path" not in df.attrs:
            subset = df.loc[(df.pickup_datetime >= pickup_start) &
                            (df.pickup_datetime < pickup_end)].copy()
            if availability == "completed":
                subset = subset.loc[(subset.dropoff_datetime < completed_before) &
                                    (subset.dropoff_datetime >= subset.pickup_datetime)]
                if completed_after is not None:
                    subset = subset.loc[subset.dropoff_datetime >= completed_after]
            return aggregate_od_flows_from_frame(
                subset, location_to_zone, set(args.excluded_zones), args.od_include_self
            )[:2]
        dataset = ds.dataset(df.attrs["od_events_path"], format="parquet")
        condition = ((ds.field("pickup_datetime") >= pickup_start.to_pydatetime()) &
                     (ds.field("pickup_datetime") < pickup_end.to_pydatetime()))
        if availability == "completed":
            condition &= ds.field("available_from") <= completed_before.to_pydatetime()
            if completed_after is not None:
                condition &= ds.field("available_from") > completed_after.to_pydatetime()
        flows, totals = Counter(), Counter()
        scanner = dataset.scanner(columns=["PULocationID", "DOLocationID", "trip_count"],
                                  filter=condition, batch_size=250_000)
        for batch in scanner.to_batches():
            partial, origins, _ = aggregate_od_flows_from_frame(
                batch.to_pandas(), location_to_zone, set(args.excluded_zones), args.od_include_self
            )
            flows.update(partial)
            totals.update(origins)
        return flows, totals

    state = states.get(key)
    if state is None or end <= state[0] or end - state[0] >= pd.Timedelta(days=lookback_days):
        flows, totals = scan(start, end, end)
    else:
        previous, previous_flows, previous_totals = state
        flows, totals = previous_flows.copy(), previous_totals.copy()
        expired, expired_totals = scan(previous-pd.Timedelta(days=lookback_days), start, previous)
        if availability == "completed":
            arriving, arriving_totals = scan(start, end, end, previous)
        else:
            arriving, arriving_totals = scan(previous, end, end)
        flows.subtract(expired)
        totals.subtract(expired_totals)
        flows.update(arriving)
        totals.update(arriving_totals)
        flows, totals = +flows, +totals
    states[key] = (end, flows.copy(), totals.copy())
    args._od_flow_states = states
    return flows, totals, int(sum(flows.values()))


def available_od_window(df, target_hour, lookback_days, availability="completed"):
    """Select destinations available strictly before the forecast instant."""
    end = pd.Timestamp(target_hour)
    start = end - pd.Timedelta(days=lookback_days)
    if "od_events_path" in df.attrs:
        if end != end.floor("h"):
            raise ValueError("Prepared OD events support hour-aligned forecasts only.")
        import pyarrow.dataset as ds

        dataset = ds.dataset(df.attrs["od_events_path"], format="parquet")
        condition = ((ds.field("pickup_datetime") >= start.to_pydatetime()) &
                     (ds.field("pickup_datetime") < end.to_pydatetime()))
        if availability == "completed":
            condition &= ds.field("available_from") <= end.to_pydatetime()
        elif availability != "pickup":
            raise ValueError(f"Unknown OD availability: {availability}")
        return dataset.to_table(
            columns=["PULocationID", "DOLocationID", "trip_count"], filter=condition
        ).to_pandas()
    mask = (df["pickup_datetime"] >= start) & (df["pickup_datetime"] < end)
    if availability == "completed":
        if "dropoff_datetime" not in df:
            raise ValueError("Completed OD graphs require dropoff_datetime.")
        mask &= (df["dropoff_datetime"] < end)
        mask &= (df["dropoff_datetime"] >= df["pickup_datetime"])
    elif availability != "pickup":
        raise ValueError(f"Unknown OD availability: {availability}")
    return df.loc[mask, ["PULocationID", "DOLocationID"]]


def build_rolling_od_graph_context(
    args: argparse.Namespace,
    df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    zone_names: Sequence[str],
    location_to_zone: Dict[int, str],
    target_hour: pd.Timestamp,
    build_graph_context_fn: Callable[[Path, pd.DataFrame], object],
) -> object:
    if "DOLocationID" not in df.columns and "od_events_path" not in df.attrs:
        raise ValueError("Rolling OD graph generation requires DOLocationID in taxi data.")

    end_time = pd.Timestamp(target_hour)
    start_time = end_time - pd.Timedelta(days=args.od_lookback_days)
    flows, origin_totals, kept_rows = aggregate_available_od(
        args, df, end_time, args.od_lookback_days, location_to_zone
    )
    retained = _retain_top_k(
        flows=flows,
        top_k=args.od_top_k,
        min_flow=args.od_min_flow,
    )
    matrix = _build_matrix(
        zone_names=list(zone_names),
        retained=retained,
        origin_totals=origin_totals,
        weight_mode=args.od_weight_mode,
        symmetrize=args.od_symmetrize,
    )

    output_path = od_graph_path(args=args, target_hour=end_time)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, encoding="utf-8")

    edge_count = int((matrix.to_numpy(dtype=float) > 0.0).sum())
    print(
        "Generated rolling OD graph "
        f"[{start_time}, {end_time}) at {output_path} "
        f"from {kept_rows} retained trips with {edge_count} nonzero edges.",
        flush=True,
    )
    return build_graph_context_fn(edge_csv=output_path, lookup_df=lookup_df)
