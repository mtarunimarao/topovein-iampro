"""
TopoVein — Persistence Homology Computation
Computes persistent homology using geodesic shortest-path distance matrices.
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from gtda.homology import VietorisRipsPersistence

def process_vein_graph(graph_path: Path, output_path: Path):
    """Computes PH for a vein skeleton graph."""
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    # Use all-pairs shortest paths as the metric for accurate vein topology
    dist_matrix = nx.floyd_warshall_numpy(G, weight='geodesic_length')

    # Handle disconnected components by capping distance
    if dist_matrix.size > 0:
        max_val = np.nanmax(dist_matrix[np.isfinite(dist_matrix)])
        dist_matrix[np.isinf(dist_matrix)] = max_val * 2.0
    
    # Giotto-tda expects (n_samples, n_nodes, n_nodes)
    X = dist_matrix[np.newaxis, :, :]

    # Vietoris-Rips transformer
    vr = VietorisRipsPersistence(
        metric="precomputed", 
        homology_dimensions=[0, 1], 
        n_jobs=-1
    )

    diagrams = vr.fit_transform(X)
    diagram = diagrams[0]
    
    # Cleaning: Remove features where birth == death
    diagram = diagram[diagram[:, 1] > diagram[:, 0]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(diagram, f)

if __name__ == "__main__":
    graph_root = Path("results/graphs")
    output_root = Path("results/tda/diagrams")
    
    files = list(graph_root.rglob("*_graph.pkl"))
    print(f"Processing {len(files)} graphs...")
    
    for p in files:
        out = output_root / f"{p.stem}_diagram.pkl"
        try:
            process_vein_graph(p, out)
        except Exception as e:
            print(f"Error on {p.name}: {e}")
            
    print("Computation complete.")