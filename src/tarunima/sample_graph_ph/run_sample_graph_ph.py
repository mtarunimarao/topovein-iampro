"""
Tarunima — Sample Graph Persistent Homology Demo

Build a hand-crafted graph, then run persistent homology in two ways:

1. Vietoris-Rips persistence on the graph shortest-path metric.
2. Vietoris-Rips persistence on the embedded node coordinate cloud.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from ph_common import analyse_graph_persistence, require_dependencies


def build_sample_graph() -> object:
    """Create a rich sample graph with two components and two cycles."""
    nx, _, _ = require_dependencies()
    graph = nx.MultiGraph()
    graph.graph["description"] = "Tarunima sample graph with two connected components and visible loops."

    positions = {
        0: (0.0, 0.0),
        1: (0.0, 2.0),
        2: (2.0, 2.0),
        3: (2.0, 0.0),
        4: (3.1, 1.0),
        5: (6.0, 0.0),
        6: (7.2, 1.2),
        7: (6.0, 2.5),
        8: (8.8, 1.2),
    }
    for node_id, (row, col) in positions.items():
        graph.add_node(
            node_id,
            row=row,
            col=col,
            pixel_row=int(round(row * 100)),
            pixel_col=int(round(col * 100)),
            kind="sample",
        )

    weighted_edges = [
        (0, 1, 1.00),
        (1, 2, 1.10),
        (2, 3, 0.95),
        (3, 0, 1.05),  # first cycle
        (2, 4, 0.80),  # spur
        (5, 6, 1.00),
        (6, 7, 0.90),
        (7, 5, 1.20),  # second cycle
        (6, 8, 1.30),  # branch off second component
    ]
    for u, v, weight in weighted_edges:
        graph.add_edge(
            u,
            v,
            weight=weight,
            geodesic_length=weight,
            euclidean_length=weight,
            branch_type="sample_edge",
        )

    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run persistent homology on a controlled sample graph.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tda/sample_graph_ph"),
        help="Directory to store sample-graph PH outputs.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_sample_graph()

    summary = analyse_graph_persistence(
        graph=graph,
        graph_name="tarunima_sample_graph",
        output_dir=args.output_dir,
        weight_attr=args.weight_attr,
        homology_dimensions=args.homology_dims,
        source_path="synthetic://tarunima_sample_graph",
        save_graph_pickle=True,
    )

    print("=" * 68)
    print("Tarunima — Sample Graph Persistent Homology")
    print("=" * 68)
    print(f"Output directory          : {args.output_dir}")
    print(f"Graph nodes / edges       : {summary['graph_statistics']['nodes']} / {summary['graph_statistics']['edges']}")
    print(f"Cycle basis count         : {summary['graph_statistics']['cycle_basis_count']}")
    print(f"Graph-metric summary JSON : {summary['outputs']['summary_json']}")
    print(f"Graph visualization       : {summary['outputs']['graph_plot']}")

    metric_summary = summary["analyses"].get("graph_metric_vr", {})
    if "features" in metric_summary:
        for dim, info in metric_summary["features"].items():
            print(
                f"H{dim} graph-metric features    : {info['count']}  "
                f"(max persistence={info['max_persistence']:.4f})"
            )

    cloud_summary = summary["analyses"].get("embedded_node_cloud_vr", {})
    if "features" in cloud_summary:
        for dim, info in cloud_summary["features"].items():
            print(
                f"H{dim} node-cloud features      : {info['count']}  "
                f"(max persistence={info['max_persistence']:.4f})"
            )


if __name__ == "__main__":
    main()
