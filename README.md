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

## Dataset
FV-USM (123 subjects, 2 sessions, 4 fingers, 6 images each)
Not included — see Dataset/README.md

## Branch Structure
- main — stable, protected, merged only via PR
- samiksha — preprocessing module
- harsh — skeletonization and graph extraction
- tarunima — TDA and matching
- rishu — hardware integration

## Tech Stack
Python · OpenCV · scikit-image · NetworkX · NumPy · Matplotlib · giotto-tda · SciPy
