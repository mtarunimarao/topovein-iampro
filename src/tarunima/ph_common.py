"""
Shared persistent-homology utilities for Tarunima's TopoVein module.

The focus here is practical, reproducible graph persistence:

- Collapse a possibly multi-edge Harsh graph into a weighted simple graph.
- Convert that graph into an all-pairs shortest-path metric space.
- Run Vietoris-Rips persistence on the graph metric.
- Optionally run Vietoris-Rips persistence on the node coordinate cloud.
- Save human-readable summaries and publication-friendly plots.
"""

from __future__ import annotations

import importlib
import json
import math
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    import networkx as nx


def require_dependencies() -> tuple[Any, Any, Any]:
    """
    Import runtime dependencies lazily so the module remains syntax-checkable
    even when scientific packages are not installed in the current shell.
    """
    missing: list[str] = []
    loaded: dict[str, Any] = {}
    for module_name in ("networkx", "matplotlib.pyplot", "gtda.homology"):
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        missing_text = ", ".join(missing)
        raise ModuleNotFoundError(
            "Missing runtime dependencies for Tarunima's persistent-homology "
            f"module: {missing_text}. Run `pip install -r requirements.txt` "
            "from the repository root."
        )

    return loaded["networkx"], loaded["matplotlib.pyplot"], loaded["gtda.homology"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_builtin(value: Any) -> Any:
    """Recursively convert NumPy scalar/container types into JSON-safe Python."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_pointer:
        pickle.dump(payload, file_pointer, protocol=pickle.HIGHEST_PROTOCOL)


def collapse_to_weighted_simple_graph(graph: Any, weight_attr: str) -> Any:
    """Collapse a NetworkX graph or MultiGraph into a simple weighted graph."""
    nx, _, _ = require_dependencies()
    simple_graph = nx.Graph()
    simple_graph.add_nodes_from(graph.nodes(data=True))

    for u, v, data in graph.edges(data=True):
        if u == v:
            continue

        weight = float(data.get(weight_attr, data.get("weight", 1.0)))
        if not math.isfinite(weight):
            continue

        if simple_graph.has_edge(u, v):
            best_weight = min(float(simple_graph[u][v][weight_attr]), weight)
            simple_graph[u][v][weight_attr] = best_weight
        else:
            simple_graph.add_edge(u, v, **{weight_attr: weight})

    return simple_graph


def extract_node_coordinates(graph: Any) -> tuple[list[Any], np.ndarray] | None:
    """
    Extract 2D node coordinates from Harsh's graph node attributes.

    Preference order:
    - `row`, `col`
    - `pixel_row`, `pixel_col`
    """
    nodes = list(graph.nodes())
    coordinates: list[list[float]] = []

    for node_id in nodes:
        attrs = graph.nodes[node_id]
        if "row" in attrs and "col" in attrs:
            coordinates.append([float(attrs["row"]), float(attrs["col"])])
        elif "pixel_row" in attrs and "pixel_col" in attrs:
            coordinates.append([float(attrs["pixel_row"]), float(attrs["pixel_col"])])
        else:
            return None

    if not coordinates:
        return None

    return nodes, np.asarray(coordinates, dtype=float)


def compute_graph_distance_matrix(
    graph: Any,
    weight_attr: str,
) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    """
    Build an all-pairs shortest-path distance matrix from a weighted graph.

    Disconnected pairs are assigned a finite cap slightly above the maximum
    observed finite distance so persistent homology can still be computed.
    """
    nx, _, _ = require_dependencies()
    simple_graph = collapse_to_weighted_simple_graph(graph, weight_attr)
    nodes = list(simple_graph.nodes())

    if not nodes:
        return [], np.zeros((0, 0), dtype=float), {
            "disconnected_pairs": 0,
            "distance_cap": 0.0,
            "max_finite_distance": 0.0,
        }

    distances = np.asarray(nx.floyd_warshall_numpy(simple_graph, nodelist=nodes, weight=weight_attr), dtype=float)

    if distances.shape[0] == 1:
        return nodes, distances, {
            "disconnected_pairs": 0,
            "distance_cap": 0.0,
            "max_finite_distance": 0.0,
        }

    finite_mask = np.isfinite(distances)
    finite_off_diag = distances[np.triu(finite_mask, k=1)]
    finite_off_diag = finite_off_diag[finite_off_diag > 0]

    max_finite = float(finite_off_diag.max()) if finite_off_diag.size else 1.0
    distance_cap = max_finite * 1.25

    disconnected_mask = ~np.isfinite(distances)
    np.fill_diagonal(disconnected_mask, False)
    disconnected_pairs = int(np.count_nonzero(np.triu(disconnected_mask, k=1)))

    distances = distances.copy()
    distances[disconnected_mask] = distance_cap
    np.fill_diagonal(distances, 0.0)

    return nodes, distances, {
        "disconnected_pairs": disconnected_pairs,
        "distance_cap": float(distance_cap),
        "max_finite_distance": float(max_finite),
    }


def run_vr_on_distance_matrix(distance_matrix: np.ndarray, homology_dimensions: Sequence[int]) -> np.ndarray:
    """Run Vietoris-Rips persistence on a precomputed distance matrix."""
    _, _, gtda_homology = require_dependencies()
    VR = gtda_homology.VietorisRipsPersistence

    transformer = VR(
        metric="precomputed",
        homology_dimensions=tuple(homology_dimensions),
        collapse_edges=True,
        n_jobs=1,
    )
    diagrams = transformer.fit_transform(distance_matrix[None, :, :])
    return diagrams[0]


def run_vr_on_point_cloud(point_cloud: np.ndarray, homology_dimensions: Sequence[int]) -> np.ndarray:
    """Run Vietoris-Rips persistence on a 2D node coordinate cloud."""
    _, _, gtda_homology = require_dependencies()
    VR = gtda_homology.VietorisRipsPersistence

    transformer = VR(
        metric="euclidean",
        homology_dimensions=tuple(homology_dimensions),
        collapse_edges=True,
        n_jobs=1,
    )
    diagrams = transformer.fit_transform(point_cloud[None, :, :])
    return diagrams[0]


def summarise_diagram(diagram: np.ndarray, finite_death_cap: float) -> dict[str, Any]:
    """Turn one persistence diagram into a compact JSON-friendly summary."""
    if diagram.size == 0:
        return {"features": {}, "total_features": 0}

    summary_by_dim: dict[str, Any] = {}
    total_features = 0

    dims = sorted({int(point[2]) for point in diagram})
    for dim in dims:
        points = diagram[diagram[:, 2] == dim]
        births = points[:, 0]
        deaths = np.where(np.isfinite(points[:, 1]), points[:, 1], finite_death_cap)
        persistences = deaths - births

        top_indices = np.argsort(-persistences)[:5]
        top_features = [
            {
                "birth": float(round(births[idx], 6)),
                "death": float(round(deaths[idx], 6)),
                "persistence": float(round(persistences[idx], 6)),
            }
            for idx in top_indices
        ]

        summary_by_dim[str(dim)] = {
            "count": int(len(points)),
            "max_persistence": float(round(float(np.max(persistences)), 6)),
            "mean_persistence": float(round(float(np.mean(persistences)), 6)),
            "top_features": top_features,
        }
        total_features += int(len(points))

    return {
        "features": summary_by_dim,
        "total_features": total_features,
    }


def plot_persistence_diagram(
    diagram: np.ndarray,
    title: str,
    output_path: Path,
    finite_death_cap: float,
) -> None:
    """Create a clean Matplotlib persistence-diagram scatter plot."""
    _, plt, _ = require_dependencies()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if diagram.size == 0:
        fig, axis = plt.subplots(figsize=(8, 6))
        axis.text(0.5, 0.5, "No persistence features found", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.axis("off")
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    plotted = diagram.copy()
    plotted[:, 1] = np.where(np.isfinite(plotted[:, 1]), plotted[:, 1], finite_death_cap)
    max_value = float(np.max(plotted[:, :2])) if plotted.size else 1.0
    max_value = max(max_value, finite_death_cap)

    fig, axis = plt.subplots(figsize=(8, 6))
    colors = {
        0: "#1d4ed8",
        1: "#dc2626",
        2: "#059669",
        3: "#7c3aed",
    }

    for dim in sorted({int(point[2]) for point in plotted}):
        points = plotted[plotted[:, 2] == dim]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            label=f"H{dim}",
            alpha=0.85,
            s=28,
            color=colors.get(dim, "#374151"),
        )

    axis.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1.2, color="#6b7280", label="birth = death")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Birth")
    axis.set_ylabel("Death")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_graph(graph: Any, output_path: Path, weight_attr: str, title: str) -> None:
    """Render a graph using embedded node coordinates when available."""
    nx, plt, _ = require_dependencies()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    simple_graph = collapse_to_weighted_simple_graph(graph, weight_attr)
    coordinates = extract_node_coordinates(simple_graph)
    if coordinates is not None:
        nodes, point_cloud = coordinates
        pos = {
            node_id: (float(point_cloud[index, 1]), -float(point_cloud[index, 0]))
            for index, node_id in enumerate(nodes)
        }
    else:
        pos = nx.spring_layout(simple_graph, seed=42, weight=weight_attr)

    node_kinds = nx.get_node_attributes(graph, "kind")
    palette = {
        "junction": "#dc2626",
        "endpoint": "#2563eb",
        "cycle": "#7c3aed",
        "isolated": "#f59e0b",
        "synthetic": "#6b7280",
    }
    node_colors = [palette.get(node_kinds.get(node_id, ""), "#111827") for node_id in simple_graph.nodes()]

    fig, axis = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(simple_graph, pos, width=1.2, alpha=0.6, edge_color="#94a3b8", ax=axis)
    nx.draw_networkx_nodes(simple_graph, pos, node_size=45, node_color=node_colors, alpha=0.95, ax=axis)
    axis.set_title(title)
    axis.axis("off")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def graph_statistics(graph: Any, weight_attr: str) -> dict[str, Any]:
    """Compute practical graph metadata for README/demo/report consumption."""
    nx, _, _ = require_dependencies()
    simple_graph = collapse_to_weighted_simple_graph(graph, weight_attr)
    weights = [float(data.get(weight_attr, 1.0)) for _, _, data in simple_graph.edges(data=True)]
    degrees = [degree for _, degree in simple_graph.degree()]

    cycle_count = 0
    if simple_graph.number_of_nodes() > 0 and simple_graph.number_of_edges() > 0:
        cycle_count = sum(len(nx.cycle_basis(simple_graph.subgraph(component).copy())) for component in nx.connected_components(simple_graph))

    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "simple_edges": int(simple_graph.number_of_edges()),
        "connected_components": int(nx.number_connected_components(simple_graph)) if simple_graph.number_of_nodes() else 0,
        "self_loops": int(nx.number_of_selfloops(graph)),
        "average_degree": float(round(float(np.mean(degrees)), 6)) if degrees else 0.0,
        "cycle_basis_count": int(cycle_count),
        "min_edge_weight": float(round(min(weights), 6)) if weights else 0.0,
        "max_edge_weight": float(round(max(weights), 6)) if weights else 0.0,
        "mean_edge_weight": float(round(float(np.mean(weights)), 6)) if weights else 0.0,
        "has_node_coordinates": bool(extract_node_coordinates(graph) is not None),
    }


def analyse_graph_persistence(
    graph: Any,
    graph_name: str,
    output_dir: Path,
    weight_attr: str = "geodesic_length",
    homology_dimensions: Sequence[int] = (0, 1, 2),
    source_path: str | None = None,
    save_graph_pickle: bool = False,
) -> dict[str, Any]:
    """
    Run the full Tarunima persistent-homology suite on one graph and save outputs.
    """
    output_dir = ensure_dir(output_dir)

    stats = graph_statistics(graph, weight_attr)
    summary: dict[str, Any] = {
        "graph_name": graph_name,
        "source_path": source_path or "",
        "weight_attr": weight_attr,
        "homology_dimensions": list(homology_dimensions),
        "graph_statistics": stats,
        "outputs": {},
        "analyses": {},
    }

    graph_png = output_dir / f"{graph_name}_graph.png"
    plot_graph(graph, graph_png, weight_attr, title=f"{graph_name} — Graph View")
    summary["outputs"]["graph_plot"] = str(graph_png)

    if save_graph_pickle:
        graph_pickle = output_dir / f"{graph_name}.pkl"
        save_pickle(graph_pickle, graph)
        summary["outputs"]["graph_pickle"] = str(graph_pickle)

    nodes, distance_matrix, metric_metadata = compute_graph_distance_matrix(graph, weight_attr)
    np.save(output_dir / f"{graph_name}_graph_metric.npy", distance_matrix)
    summary["outputs"]["graph_metric_matrix"] = str(output_dir / f"{graph_name}_graph_metric.npy")
    summary["graph_metric"] = {
        "node_count": len(nodes),
        **metric_metadata,
    }

    if len(nodes) >= 2:
        metric_diagram = run_vr_on_distance_matrix(distance_matrix, homology_dimensions)
        np.save(output_dir / f"{graph_name}_graph_metric_diagram.npy", metric_diagram)
        metric_cap = max(metric_metadata["distance_cap"], metric_metadata["max_finite_distance"], 1.0)
        metric_plot = output_dir / f"{graph_name}_graph_metric_diagram.png"
        plot_persistence_diagram(
            metric_diagram,
            title=f"{graph_name} — PH on Graph Shortest-Path Metric",
            output_path=metric_plot,
            finite_death_cap=float(metric_cap),
        )
        summary["outputs"]["graph_metric_diagram"] = str(output_dir / f"{graph_name}_graph_metric_diagram.npy")
        summary["outputs"]["graph_metric_plot"] = str(metric_plot)
        summary["analyses"]["graph_metric_vr"] = summarise_diagram(metric_diagram, finite_death_cap=float(metric_cap))
    else:
        summary["analyses"]["graph_metric_vr"] = {"note": "Need at least 2 nodes for Vietoris-Rips persistence."}

    coordinate_bundle = extract_node_coordinates(graph)
    if coordinate_bundle is not None and len(coordinate_bundle[0]) >= 2:
        _, point_cloud = coordinate_bundle
        np.save(output_dir / f"{graph_name}_node_coordinates.npy", point_cloud)
        euclidean_diagram = run_vr_on_point_cloud(point_cloud, homology_dimensions)
        np.save(output_dir / f"{graph_name}_node_cloud_diagram.npy", euclidean_diagram)

        finite_values = euclidean_diagram[:, :2][np.isfinite(euclidean_diagram[:, :2])]
        euclidean_cap = float(np.max(finite_values)) * 1.1 if finite_values.size else 1.0
        euclidean_plot = output_dir / f"{graph_name}_node_cloud_diagram.png"
        plot_persistence_diagram(
            euclidean_diagram,
            title=f"{graph_name} — PH on Embedded Node Cloud",
            output_path=euclidean_plot,
            finite_death_cap=max(euclidean_cap, 1.0),
        )

        summary["outputs"]["node_coordinates"] = str(output_dir / f"{graph_name}_node_coordinates.npy")
        summary["outputs"]["node_cloud_diagram"] = str(output_dir / f"{graph_name}_node_cloud_diagram.npy")
        summary["outputs"]["node_cloud_plot"] = str(euclidean_plot)
        summary["analyses"]["embedded_node_cloud_vr"] = summarise_diagram(
            euclidean_diagram,
            finite_death_cap=max(euclidean_cap, 1.0),
        )
    else:
        summary["analyses"]["embedded_node_cloud_vr"] = {
            "note": "Node coordinate attributes were missing or insufficient for coordinate-cloud persistence."
        }

    summary_path = output_dir / f"{graph_name}_summary.json"
    summary["outputs"]["summary_json"] = str(summary_path)
    save_json(summary_path, summary)
    return summary
