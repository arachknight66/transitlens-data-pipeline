"""
datasets package
─────────────────
Assembles labelled light curve datasets for transitlens-ml-core.

Public API:
    build_from_synthetic()  — builds labeled_dataset.csv from synthetic cases
    split_dataset()         — creates train/val/test splits by target_id
"""

from datasets.build_dataset import build_from_synthetic, split_dataset