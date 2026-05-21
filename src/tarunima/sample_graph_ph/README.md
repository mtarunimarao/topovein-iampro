# Tarunima — Sample Graph Persistent Homology

This folder contains a self-contained demonstration of persistent homology on a
controlled graph with:

- one disconnected component pair for strong `H0` behavior
- multiple cycles for non-trivial `H1`
- embedded node coordinates for a second spatial persistence view

## Why this exists

Before running topology analysis on real Harsh-exported graphs, this script lets
the team validate the persistent-homology workflow on a graph whose structure is
easy to reason about.

## Run

From the repository root:

```bash
python src/tarunima/sample_graph_ph/run_sample_graph_ph.py
```

## Outputs

The script writes artifacts to:

```text
results/tda/sample_graph_ph/
```

Typical outputs include:

- the sample graph pickle
- a graph visualization PNG
- a graph-metric distance matrix `.npy`
- persistence diagrams `.npy`
- persistence diagram plots `.png`
- a summary JSON with feature counts, strongest classes, and graph stats
