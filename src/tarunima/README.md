# Tarunima Module — Persistent Homology

This folder contains Tarunima's topology-analysis layer for the TopoVein
pipeline. It is split into two push-ready subfolders:

- `sample_graph_ph/`
  Applies persistent homology to a hand-crafted sample graph so the team can
  verify the methodology on a controlled example with known cycles and
  connected components.

- `harsh_graph_ph/`
  Applies persistent homology to the `networkx.MultiGraph` files exported by
  Harsh's graph-extraction pipeline in `results/graphs/`.

Shared graph-to-persistence utilities live in `ph_common.py`.

## Method Used

Both scripts support a graph-metric persistent-homology workflow:

1. Collapse the graph to a weighted simple graph.
2. Build an all-pairs shortest-path distance matrix.
3. Run Vietoris-Rips persistent homology on that precomputed graph metric.
4. Optionally also run Vietoris-Rips persistent homology on the node
   coordinate cloud if graph nodes expose `(row, col)` geometry.

This keeps the code compatible with the existing stack in `requirements.txt`,
especially `giotto-tda`, `networkx`, `numpy`, and `matplotlib`.

## Expected Dependencies

Install the project requirements first:

```bash
pip install -r requirements.txt
```

On Windows, use:

```bash
python -m pip install -r requirements-tarunima.txt
```

That lighter file is enough for Tarunima's two scripts because they only need
`numpy`, `networkx`, `matplotlib`, and `giotto-tda`.

## Quick Start

Sample graph demo:

```bash
python src/tarunima/sample_graph_ph/run_sample_graph_ph.py
```

Harsh graph demo:

```bash
python src/tarunima/harsh_graph_ph/run_harsh_graph_ph.py --limit 5
```

If the Harsh-graph script says no `*_graph.pkl` files were found, that means
the PH layer is installed correctly but Harsh's graph-export step has not been
run yet or the graph folder is somewhere else. In that case either:

- run `src/graph_builder/06_graph.py` first, or
- pass the real graph location with `--graph-root`.
