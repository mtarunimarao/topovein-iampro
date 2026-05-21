# TopoVein — IEEE IAMPro 2026

Topology-based biometric authentication using finger vein graph analysis
and persistent homology.

**Team:** Central University of Karnataka  
**Program:** IEEE CS Bangalore Chapter Internship 2026

## Pipeline

NIR Image → Preprocessing → Skeletonization → Graph G(V,E) → Persistent Homology → Hausdorff Matching → Auth Decision

## Setup

```bash
pip install -r requirements.txt
```

Windows users can safely use:

```bash
python -m pip install -r requirements.txt
```

If you only want Tarunima's persistent-homology scripts, install the lighter
dependency set instead:

```bash
python -m pip install -r requirements-tarunima.txt
```

## Dataset
This repo can work with:

- the official FV-USM dataset placed in `Dataset/Published_database_FV-USM_Dec2013`
- the UTFVP dataset placed in `Dataset/Published_database_UTFVP`

This repository does not ship biometric image data. Add one of the supported
datasets locally before running the preprocessing, skeletonization, graph
extraction, and Tarunima PH pipeline.

## Branch Structure
- main — stable, protected, merged only via PR
- samiksha — preprocessing module
- harsh — skeletonization and graph extraction
- tarunima — TDA and matching
- rishu — hardware integration

## Tech Stack
Python · OpenCV · scikit-image · NetworkX · NumPy · Matplotlib · giotto-tda · SciPy
