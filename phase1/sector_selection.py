"""Deterministic sector ranking from a frozen inventory."""

import pandas as pd


def select_sectors(config, success_margin=0.10):
    inventory_path = config.manifests_dir / "sector_inventory.parquet"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Sector inventory not found: {inventory_path}")
    inventory = pd.read_parquet(inventory_path)
    required = int(config.minimum_successful_observations / (1.0 - success_margin))
    ranked = inventory.sort_values(
        ["n_observations", "sector"], ascending=[False, True]
    ).copy()
    ranked["selected"] = False
    cumulative = 0
    for index, row in ranked.iterrows():
        if cumulative >= required:
            break
        ranked.at[index, "selected"] = True
        cumulative += int(row["n_observations"])
    ranked["selection_target"] = required
    ranked["cumulative_selected_observations"] = cumulative
    ranked.to_parquet(config.manifests_dir / "sector_selection.parquet", index=False)
    if cumulative < required:
        raise RuntimeError(
            f"Frozen inventory contains {cumulative} eligible observations; "
            f"{required} are required after the configured safety margin."
        )
    return ranked[ranked["selected"]].copy()
