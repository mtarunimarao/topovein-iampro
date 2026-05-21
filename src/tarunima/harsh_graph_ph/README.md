# Tarunima — Persistent Homology on Harsh Graph Exports

This folder applies persistent homology directly to the graph pickles generated
by Harsh's `src/graph_builder/06_graph.py`.

## Input Expected

Harsh's pipeline writes graphs to:

```text
results/graphs/<session>/<subject_finger>/<image>_graph.pkl
```

Each pickle should contain a `networkx.MultiGraph` with node geometry such as
`row`, `col`, `pixel_row`, `pixel_col`, and edge geometry such as
`geodesic_length`.

## What this script does

For each graph:

1. Loads the graph pickle.
2. Collapses it to a weighted simple graph for graph-metric analysis.
3. Builds an all-pairs shortest-path matrix from `geodesic_length`.
4. Runs Vietoris-Rips persistence on that graph metric.
5. If node coordinates exist, runs a second persistence pass on the embedded
   node cloud.
6. Saves diagrams, plots, and a summary JSON.

## Run One Graph

```bash
python src/tarunima/harsh_graph_ph/run_harsh_graph_ph.py \
  --graph-pkl results/graphs/1st_session/001_1/01_graph.pkl
```

## Run a Batch

```bash
python src/tarunima/harsh_graph_ph/run_harsh_graph_ph.py --limit 20
```

## Outputs

Artifacts are written under:

```text
results/tda/harsh_graph_ph/
```

The script mirrors the input session/subject structure so outputs remain easy to
trace back to the original graph export.
