"""
Script 01: Organize Raw Datasets.

Verifies downloads, organizes into processed directory structure,
performs deduplication, and creates train/val splits.

Usage:
    python scripts/01_organize_data.py                 # Full organization
    python scripts/01_organize_data.py --verify-only   # Just verify downloads
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config import get_config
from src.data.organizer import verify_raw_data, organize_all


def main():
    parser = argparse.ArgumentParser(description="Organize raw datasets")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify downloads, don't organize"
    )
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="Skip deduplication step (faster)"
    )
    args = parser.parse_args()

    config = get_config()

    if args.verify_only:
        print("Running verification only...\n")
        verify_raw_data(config.data.raw_dir)
        return

    print("=" * 60)
    print("  DATASET ORGANIZATION PIPELINE")
    print("=" * 60)

    organize_all(
        raw_dir=config.data.raw_dir,
        processed_dir=config.data.processed_dir,
        severity_dir=config.data.severity_dir,
        quarantine_dir=config.data.quarantine_dir,
        train_ratio=config.data.train_val_split,
        phash_threshold=config.data.phash_threshold if not args.skip_dedup else 0,
        seed=config.data.seed,
    )


if __name__ == "__main__":
    main()
