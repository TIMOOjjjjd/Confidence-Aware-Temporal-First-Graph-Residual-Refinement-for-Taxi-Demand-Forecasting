"""Extract a small immutable no-graph forecast cache, without any TCN execution."""

import argparse
import json
from pathlib import Path

from experiment_2_from_tcn import load_no_graph_tcn_baseline
from revision_protocol import sha256_file


def freeze(source, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen cache: {output}")
    frame = load_no_graph_tcn_baseline(source, "base_pred").reset_index()
    frame["graph_type"] = "no_graph"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "baseline.csv"
    frame.to_csv(cache_path, index=False, mode="x")
    manifest = {"source": str(source.resolve()), "source_sha256": sha256_file(source),
                "cache": str(cache_path.resolve()), "cache_sha256": sha256_file(cache_path),
                "rows": len(frame), "hours": frame.target_hour.nunique(),
                "nodes": frame.PULocationID.nunique(), "tcn_execution": "none"}
    (output / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.source, args.output_dir)
