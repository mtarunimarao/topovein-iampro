"""
Tarunima — Persistent Homology on Harsh Graph Exports

Batch-ready script for applying persistent homology to the graph pickle files
created by `src/graph_builder/06_graph.py`.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from ph_common import analyse_graph_persistence, ensure_dir, save_json, require_dependencies


def load_graph_pickle(path: Path) -> Any:
    with path.open("rb") as file_pointer:
        return pickle.load(file_pointer)


def discover_graph_pickles(graph_root: Path, limit: int | None = None) -> list[Path]:
    graph_paths = sorted(graph_root.glob("**/*_graph.pkl"))
    if limit is not None:
        graph_paths = graph_paths[:limit]
    return graph_paths


def build_output_dir_for_graph(graph_pkl: Path, graph_root: Path, output_root: Path) -> Path:
    relative_parent = graph_pkl.relative_to(graph_root).parent
    return output_root / relative_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run persistent homology on Harsh-exported graph pickles.")
    parser.add_argument(
        "--graph-pkl",
        type=Path,
        default=None,
        help="Analyze one specific graph pickle.",
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("results/graphs"),
        help="Root directory containing Harsh's graph pickles.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/tda/harsh_graph_ph"),
        help="Root directory for Tarunima's persistent-homology outputs.",
    )
    parser.add_argument(
        "--weight-attr",
        default="geodesic_length",
        help="Edge attribute to use as graph-metric distance.",
    )
    parser.add_argument(
        "--homology-dims",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Homology dimensions to compute.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When scanning a graph root, process only the first N graph pickles.",
    )
    return parser.parse_args()


def main() -> None:
    require_dependencies()
    args = parse_args()
    output_root = ensure_dir(args.output_root)

    if args.graph_pkl is not None:
        graph_paths = [args.graph_pkl]
        graph_root = args.graph_pkl.parent.parent.parent if args.graph_pkl.name.endswith("_graph.pkl") else args.graph_root
    else:
        graph_root = args.graph_root
        graph_paths = discover_graph_pickles(graph_root, limit=args.limit)

    if not graph_paths:
        raise FileNotFoundError(
            "No *_graph.pkl files found.\n"
            f"Checked graph root: {graph_root}\n"
            "What this means:\n"
            "- Tarunima's persistent-homology code is ready, but Harsh's exported graphs are not present there.\n"
            "- Run src/graph_builder/06_graph.py first, or point this script at the correct folder with --graph-root.\n"
            "Example:\n"
            "python src/tarunima/harsh_graph_ph/run_harsh_graph_ph.py --graph-root results/graphs --limit 5"
        )

    batch_manifest: dict[str, Any] = {
        "graph_root": str(graph_root),
        "output_root": str(output_root),
        "weight_attr": args.weight_attr,
        "homology_dimensions": args.homology_dims,
        "graphs_processed": [],
    }

    print("=" * 72)
    print("Tarunima — Persistent Homology on Harsh Graph Exports")
    print("=" * 72)
    print(f"Graphs queued            : {len(graph_paths)}")
    print(f"Graph root               : {graph_root}")
    print(f"Output root              : {output_root}")
    print(f"Weight attribute         : {args.weight_attr}")
    print()

    for index, graph_pkl in enumerate(graph_paths, start=1):
        graph = load_graph_pickle(graph_pkl)
        graph_name = graph_pkl.stem
        target_dir = build_output_dir_for_graph(graph_pkl, graph_root, output_root) if graph_root in graph_pkl.parents else output_root / graph_name

        summary = analyse_graph_persistence(
            graph=graph,
            graph_name=graph_name,
            output_dir=target_dir,
            weight_attr=args.weight_attr,
            homology_dimensions=args.homology_dims,
            source_path=str(graph_pkl),
            save_graph_pickle=False,
        )
        batch_manifest["graphs_processed"].append(summary)

        graph_stats = summary["graph_statistics"]
        metric_summary = summary["analyses"].get("graph_metric_vr", {})
        h1_info = metric_summary.get("features", {}).get("1", {}) if isinstance(metric_summary, dict) else {}
        h1_count = h1_info.get("count", 0)
        h1_max = h1_info.get("max_persistence", 0.0)

        print(
            f"[{index:>3}/{len(graph_paths)}] {graph_pkl.name}  "
            f"nodes={graph_stats['nodes']:>4}  edges={graph_stats['edges']:>4}  "
            f"H1={h1_count:>3}  maxH1={h1_max}"
        )

    manifest_path = output_root / "batch_manifest.json"
    save_json(manifest_path, batch_manifest)

    print()
    print(f"Batch manifest           : {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
