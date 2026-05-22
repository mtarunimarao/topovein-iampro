"""
TopoVein — Betti Feature Extraction
Aggregates persistence diagrams into a summary CSV for biometric matching.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def extract_betti_summary():
    diagram_files = list(Path("results/tda/diagrams").rglob("*_diagram.pkl"))
    results = []

    for df in diagram_files:
        with open(df, 'rb') as f:
            diag = pickle.load(f)
            
            # Beta0: Connected components (count of points in dim 0)
            beta0 = np.sum(diag[:, 2] == 0)
            # Beta1: Loops/Vein crossings (count of points in dim 1)
            beta1 = np.sum(diag[:, 2] == 1)
            
            results.append({
                "filename": df.name,
                "beta0": int(beta0),
                "beta1": int(beta1),
                "total_features": int(diag.shape[0])
            })

    # Save to CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv("results/tda/betti_features.csv", index=False)
    print("Betti features saved to results/tda/betti_features.csv")
    print(df_out.head())

if __name__ == "__main__":
    extract_betti_summary()