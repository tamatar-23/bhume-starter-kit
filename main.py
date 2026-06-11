#!/usr/bin/env python3
"""CLI interface to run the BhuMe land-boundary correction pipeline.

Usage:
    python main.py data/Vadnerbhairav
"""

import sys
import argparse
from pathlib import Path

from bhume import load, score
from bhume.corrector import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="BhuMe Land-Boundary Correction Pipeline")
    parser.add_argument(
        "village_dir",
        type=str,
        help="Path to the village directory containing input.geojson, imagery.tif, boundaries.tif"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of plots to sample for global shift estimation (default: 100)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.15,
        help="Confidence threshold for correcting vs flagging plots (default: 0.15)"
    )
    parser.add_argument(
        "--sigma-reg",
        type=float,
        default=8.0,
        help="Local Gaussian prior regularization standard deviation in meters (default: 8.0)"
    )
    args = parser.parse_args()
    
    village_dir = Path(args.village_dir)
    if not village_dir.exists():
        print(f"Error: Village directory {village_dir} does not exist.")
        sys.exit(1)
        
    print(f"Running boundary correction pipeline on: {village_dir.name}...")
    
    # Run the pipeline
    preds, (global_dx, global_dy) = run_pipeline(
        village_dir=village_dir,
        sample_size=args.sample_size,
        conf_threshold=args.conf_threshold,
        sigma_reg=args.sigma_reg
    )
    
    # Reload the village to score predictions if example_truths is available
    village = load(village_dir)
    if village.example_truths is not None:
        print("\nEvaluating predictions against example truths...")
        scorecard = score(preds, village)
        print(scorecard)
    else:
        print("\nPredictions generated successfully (no example truths available to self-score).")
        print(f"Wrote predictions to {village_dir / 'predictions.geojson'}")

if __name__ == '__main__':
    main()
