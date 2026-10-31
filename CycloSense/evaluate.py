#!/usr/bin/env python3
"""
CycloSense — Live Model Evaluation CLI Entry Point
===================================================
Loads actual model weights and runs REAL inference every time.
Run from the project root:

  python evaluate.py                                        # All models, live inference
  python evaluate.py --model 2                             # Model 2 only
  python evaluate.py --model 2 --m2-csv labels.csv --m2-img-dir imgs/  # Full eval
  python evaluate.py --m1-data path/to/data.yaml           # YOLO real mAP
  python evaluate.py --m3-data ibtracs.csv                 # LSTM real errors
  python evaluate.py --json                                 # JSON output
  python evaluate.py --export results.json                  # Save to file
"""

import os
import sys

# Ensure ml module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.src.evaluation.evaluate_models import main

if __name__ == "__main__":
    main()
