"""Small deterministic controls; never load/train the temporal predictor."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from gnn_model import make_residual_model
from prepare_revision_data import aggregate_batch, prepare, load_cached_trip_data
from rolling_od_graph_utils import (
    available_od_window, aggregate_od_flows_from_frame, aggregate_available_od,
)
from experiment_2_from_tcn import load_no_graph_tcn_baseline
from revision_protocol import add_revision_args, start_revision_run, freeze_cohort
from analyze_revision_predictions import match_predictions, analyze as analyze_predictions
from analyze_confidence_controls import historical_scale


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph_data():
    return Data(x=torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10,
                base_pred=torch.ones(4), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                edge_weight=torch.ones(3), confidence_feature_index=1)


@pytest.mark.parametrize("kind", ["mlp", "node_sage"])
def test_controls_do_not_propagate_neighbour_changes(kind):
    torch.manual_seed(7)
    data = graph_data()
    cfg = SimpleNamespace(hidden_dim=8, dropout=0.0, model_type=kind,
                          use_confidence_feature=True)
    model = make_residual_model(data, cfg).eval()
    before = model(data)[0].detach()
    changed = data.clone()
    changed.x[0] += 100
    after = model(changed)[0].detach()
    torch.testing.assert_close(before[1:], after[1:])


def test_excluded_confidence_has_no_effect_and_does_not_mutate_input():
    data = graph_data()
    cfg = SimpleNamespace(hidden_dim=8, dropout=0.0, model_type="graphsage",
                          use_confidence_feature=False)
    model = make_residual_model(data, cfg).eval()
    before = model(data)[0]
    changed = data.clone()
    changed.x[:, 1] += 1000
    torch.testing.assert_close(before, model(changed)[0])
    assert data.x.shape[1] == 5


def test_completed_od_excludes_unknown_invalid_and_boundary_destinations(tmp_path):
    frame = pd.DataFrame({
        "pickup_datetime": pd.to_datetime(["2021-03-01 05:01"] * 5),
        "dropoff_datetime": pd.to_datetime([
            "2021-03-01 05:59", "2021-03-01 06:00", "2021-03-01 06:01",
            None, "2021-03-01 04:00",
        ]), "PULocationID": [1] * 5, "DOLocationID": [2] * 5,
    })
    raw = available_od_window(frame, "2021-03-01 06:00", 1)
    assert len(raw) == 1
    source = tmp_path / "raw.parquet"
    frame.to_parquet(source)
    output = tmp_path / "prepared"
    prepare(source, output, batch_size=2)
    cached = load_cached_trip_data(output, True)
    prepared = available_od_window(cached, "2021-03-01 06:00", 1)
    assert prepared.trip_count.sum() == len(raw)
    flows, totals, rows = aggregate_od_flows_from_frame(prepared, {1: "a", 2: "b"}, set(), False)
    assert flows[("a", "b")] == totals["a"] == rows == 1
    assert cached.trip_count.sum() == len(frame)
    legacy = available_od_window(cached, "2021-03-01 06:00", 1, "pickup")
    assert legacy.trip_count.sum() == len(frame)


def test_cache_rejects_conflicting_duplicates(tmp_path):
    frame = pd.DataFrame({"target_hour": ["2021-03-01"] * 2,
                          "graph_type": ["no_graph"] * 2, "PULocationID": [1] * 2,
                          "true_value": [4] * 2, "base_pred": [3, 7]})
    path = tmp_path / "cache.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        load_no_graph_tcn_baseline(path, "base_pred")
    frame.base_pred = 3
    frame.to_csv(path, index=False)
    assert len(load_no_graph_tcn_baseline(path, "base_pred")) == 1


def test_manifest_and_cohort_protect_outputs(tmp_path):
    parser = argparse.ArgumentParser()
    add_revision_args(parser)
    args = parser.parse_args([])
    args.seed = 9
    args.gnn_epochs = args.gnn_hidden_dim = args.gnn_patience = 1
    args.train_ratio, args.val_ratio = 0.6, 0.2
    args.rolling_steps = 1
    args.prediction_column = "base_pred"
    args.data = tmp_path / "raw.parquet"
    args.tcn_detail = tmp_path / "cache.csv"
    args.tcn_detail.write_text("frozen", encoding="utf-8")
    args.results_dir = tmp_path / "results"
    start_revision_run(args, __file__)
    assert args.model_seed == 9 and args.split_seed == 42
    index = pd.MultiIndex.from_product([[pd.Timestamp("2021-03-01")], [1, 2, 3]],
                                      names=["target_hour", "PULocationID"])
    cache = pd.DataFrame({"base_pred": 1., "true_value": 2.}, index=index)
    assert freeze_cohort(args, [1, 2, 3], cache, [pd.Timestamp("2021-03-01")]) == [1, 2, 3]
    with pytest.raises(FileExistsError):
        start_revision_run(args, __file__)
    cache.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        freeze_cohort(args, [1, 2, 3], cache, [pd.Timestamp("2021-03-01")])


@pytest.mark.parametrize("availability", ["completed", "pickup"])
@pytest.mark.parametrize("prepared", [False, True])
def test_incremental_od_matches_full_rebuild(tmp_path, availability, prepared):
    times = pd.date_range("2021-02-28", periods=80, freq="h")
    frame = pd.DataFrame({"pickup_datetime": times,
                          "dropoff_datetime": times + pd.Timedelta(minutes=90),
                          "PULocationID": [1]*len(times), "DOLocationID": [2]*len(times)})
    frame.loc[3, "dropoff_datetime"] = pd.NaT
    if prepared:
        source = tmp_path / "source.parquet"
        frame.to_parquet(source)
        prepare(source, tmp_path / "prepared", 13)
        frame = load_cached_trip_data(tmp_path / "prepared", True)
    args = SimpleNamespace(od_availability=availability, excluded_zones=[], od_include_self=False)
    mapping = {1: "a", 2: "b"}
    for hour in ["2021-03-01 00:00", "2021-03-01 01:00", "2021-03-01 07:00",
                 "2021-03-02 03:00"]:
        actual = aggregate_available_od(args, frame, hour, 1, mapping)
        expected = aggregate_od_flows_from_frame(
            available_od_window(frame, hour, 1, availability), mapping, set(), False
        )
        assert actual == expected


def test_mlp_capacity_matches_active_graphsage_budget():
    data = graph_data()
    cfg = SimpleNamespace(hidden_dim=32, dropout=0., model_type="mlp",
                          use_confidence_feature=True)
    mlp = make_residual_model(data, cfg)
    actual = sum(parameter.numel() for parameter in mlp.parameters())
    expected = 2*5*32 + 2*32*32 + 3*32 + 1
    assert abs(actual-expected)/expected < .03


def test_matching_and_seed_summary_do_not_confuse_hour_variance():
    rows = []
    for seed in [11, 42]:
        for hour in pd.date_range("2021-03-01", periods=3, freq="D"):
            for variant in ["MLP", "GNN"]:
                rows.append({"model_seed": seed, "target_hour": hour, "PULocationID": 1,
                             "variant": variant, "true_value": 10., "base_pred": 8.,
                             "refined_pred": 9., "raw_residual_pred": 1.,
                             "history_snapshots": 1})
    matched, coverage = match_predictions(pd.DataFrame(rows))
    _, per_seed, summary, intervals = analyze_predictions(matched, [1.], [6], 20)
    assert coverage.excluded_rows.sum() == 0
    assert summary.seed_count.eq(2).all()
    assert summary.MAE_seed_std.eq(0).all()
    assert intervals.day_blocks.eq(3).all()
    _, _, _, graph_intervals = analyze_predictions(
        matched, [1.], [6], 20, baseline="MLP"
    )
    assert graph_intervals.baseline.eq("MLP").all()
    assert graph_intervals.loc[graph_intervals.variant.eq("GNN"), "MAE_delta"].iloc[0] == 0.
    with pytest.raises(ValueError, match="baseline not present"):
        analyze_predictions(matched, [1.], [6], 20, baseline="missing")


def test_demand_scale_does_not_use_target_label():
    index = pd.MultiIndex.from_product([[1], pd.date_range("2021-03-01", periods=169, freq="h")],
                                      names=["PULocationID", "datetime"])
    counts = pd.Series(2., index=index)
    counts.iloc[-1] = 100000
    target = pd.DataFrame({"target_hour": [pd.Timestamp("2021-03-08")], "PULocationID": [1]})
    scale = historical_scale(target, counts)
    assert scale.historical_mean.iloc[0] == 2.


@pytest.mark.parametrize("mode", ["none", "full", "learned_softmax"])
def test_target_labels_do_not_change_residual_training_or_inference(mode):
    from experiment_3_confidence_weighted_gnn_ablation import (
        GNNTrainingConfig, SplitMasks, train_residual_gnn_with_weighted_loss,
    )

    train = graph_data()
    train.y = torch.tensor([1., -2., 1., 4.])
    train.true_value = train.base_pred + train.y
    for name in ["prior_score", "stability_score", "history_consistency_score", "full_confidence"]:
        setattr(train, name, torch.full((4,), .8))
    inference = train.clone()
    masks = SplitMasks(train=torch.tensor([True, True, False, False]),
                       val=torch.tensor([False, False, True, False]),
                       test=torch.tensor([False, False, False, True]))
    cfg = GNNTrainingConfig(hidden_dim=8, dropout=0., epochs=3, patience=3)
    before = train_residual_gnn_with_weighted_loss(
        train.clone(), masks, inference, mode, torch.device("cpu"), cfg, 42
    )
    poisoned = inference.clone()
    poisoned.y += 100000
    poisoned.true_value += 100000
    after = train_residual_gnn_with_weighted_loss(
        train.clone(), masks, poisoned, mode, torch.device("cpu"), cfg, 42
    )
    np.testing.assert_allclose(before.refined_pred, after.refined_pred)
    assert before.best_epoch == after.best_epoch
