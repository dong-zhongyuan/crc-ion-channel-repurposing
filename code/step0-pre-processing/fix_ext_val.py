#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix external validation data format for figure3.

Original file format: row0='sample', row1='label', row2+=expression
New format: row0='sample', row1='label', row2='dataset', row3+=expression

This script inserts a dataset row between the label row and expression data.
"""

import csv
import os
import sys
from pathlib import Path

# Paths
_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
INPUT_FILE = _PROJECT_ROOT / "data" / "external_validate_data.csv"
OUTPUT_FILE = _PROJECT_ROOT / "data" / "external_validation_data.csv"


def main():
    """Add dataset row to external validation data."""
    # Read original file
    with open(INPUT_FILE, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    print(f"Read {len(rows)} rows from {INPUT_FILE}")

    # Extract dataset names from sample IDs (GSE prefix)
    sample_ids = rows[0][1:]  # Skip 'sample' header
    datasets = [str(s).split("_")[0] for s in sample_ids]

    print(f"Datasets detected: {list(set(datasets))}")

    # Insert dataset row (row2) between label row (row1) and expression data (row3+)
    new_rows = []

    # Row 0: sample row
    new_rows.append(rows[0])

    # Row 1: label row
    new_rows.append(rows[1])

    # Row 2: dataset row
    dataset_row = ["dataset"] + datasets
    new_rows.append(dataset_row)

    # Row 3+: expression data
    new_rows.extend(rows[2:])

    # Write output
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"Wrote {len(new_rows)} rows to {OUTPUT_FILE}")
    print(f"Row 0: {new_rows[0][:3]}")
    print(f"Row 1: {new_rows[1][:3]}")
    print(f"Row 2: {new_rows[2][:3]}")
    print(f"Row 3: {new_rows[3][:3]}")


if __name__ == "__main__":
    main()
