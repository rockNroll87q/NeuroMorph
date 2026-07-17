#!/usr/bin/env python3
"""
=============================================================================
Connor Dalby
Austin Dibble
University of Glasgow
2026

scripts/get_test_data.py

Utility to fetch test data for debugging NeuroMorph inference (adapted from NeuroMorph).

If using the template mode, this script requires templateflow.
To avoid dependency conflicts with the other repo packages, install as: 

    pip install templateflow "numpy<=1.24.3" "typing-extensions<4.6.0"

Two modes:
  --mode template   Download the MNI152 1mm skull-stripped T1w template via
                    TemplateFlow. Real data, correct space, meaningful outputs.
                    Requires internet access (~8MB download, cached after first run).

  --mode synthetic  Generate a synthetic skull-shaped volume using nibabel and numpy. 
                    No internet required. Useful for pure pipeline/shape
                    testing, but model outputs will be arbitrary.

Examples
--------
  python scripts/get_test_data.py --output ./test_data/
  python scripts/get_test_data.py --output ./test_data/ --mode synthetic
  python scripts/get_test_data.py --output ./test_data/ --mode template -n 3
  
=============================================================================
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Get our argument parser"""
    parser = argparse.ArgumentParser(
        description="Fetch or generate test NIfTI data for NeuroMorph debugging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        default="./test_data/",
        help="Directory to write test volumes into (default: ./test_data/).",
    )
    
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=1,
        help="Number of volumes to produce. For 'real' mode, copies of the "
             "same template are written with different filenames (useful for "
             "testing batch/CSV input). Default: 1.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Real T1w Model Inference Examples
# ---------------------------------------------------------------------------
def fetch_real_t1w(output_dir: str, n: int) -> List[str]:
    """
    Logic Summary
        Downloads only the raw anatomical T1w file for n subjects from the
        AOMIC PIOP2 dataset on OpenNeuro using openneuro-py. The include
        filter restricts the download to a single small file per subject,
        avoiding the diffusion and functional data that make up most of
        the dataset size. Data is freely available with no application or
        account required.

    Args
        output_dir. Directory to save the downloaded files.
        n. Number of subject scans to download.

    Returns
        List of file paths for the downloaded T1w volumes.
    """
    try:
        import openneuro
    except ImportError:
        print(
            'openneuro-py not installed. Install with\n'
            '  pip install openneuro-py'
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "_tmp_download")
    dataset = "ds002790"
    subjects = [f"sub-{i + 1:04d}" for i in range(n)]
    include = [f"{s}/anat/{s}_T1w.nii.gz" for s in subjects]

    print(f"Fetching {n} real T1w scans from AOMIC dataset {dataset}...")
    openneuro.download(dataset=dataset, target_dir=tmp_dir, include=include)

    written = []
    try:
        for s in subjects:
            src = os.path.join(tmp_dir, s, "anat", f"{s}_T1w.nii.gz")
            dest = os.path.join(output_dir, f"{s}_T1w.nii.gz")
            if not os.path.exists(src):
                anat_dir = os.path.dirname(src)
                found = os.listdir(anat_dir) if os.path.isdir(anat_dir) else "directory not found"
                print(f"  Missing {src}. Actual contents: {found}")
                continue
            shutil.move(src, dest)
            written.append(dest)
            print(f"  Downloaded {dest}")
    except OSError as exc:
        print(f"  Reorganisation failed: {exc!r}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return written


# ---------------------------------------------------------------------------
# CSV helper - optional, generates a subjects.csv for batch testing
# ---------------------------------------------------------------------------

def write_test_csv(paths: List[str], output_dir: str) -> str:
    """Write a subjects.csv pointing at the generated test volumes."""
    import pandas as pd
    csv_path = os.path.join(output_dir, "subjects.csv")
    df = pd.DataFrame({"T1": paths, "Subject": [f"sub-{i+1:02d}" for i in range(len(paths))]})
    df.to_csv(csv_path, index=False)
    print(f"  CSV written: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main"""
    args = parse_args()

    print("\nNeuroMorph test data utility")
    print(f"  Output : {args.output}")
    print(f"  N      : {args.n}\n")

    paths = fetch_real_t1w(args.output, args.n)

    if args.n > 1:
        csv_path = write_test_csv(paths, args.output)

    print("\nDone. Test with:")
    if args.n > 1:
            print(f"  python scripts/run_inference.py --input {csv_path} --output {args.output}")

if __name__ == "__main__":
    main()