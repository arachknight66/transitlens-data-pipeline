"""
synthetic/generator.py
──────────────────────
Generates synthetic TESS-like light curves from config.yaml.

This is the main orchestrator for Phase 1. It:
1. Creates a time array with realistic TESS cadence and data gaps
2. Creates a flat baseline flux array
3. Delegates to noise_models.py for noise injection
4. Delegates to transit_injector.py for transit signal injection
5. Writes output CSVs to synthetic/cases/

Performance note
-----------------
`generate_all_cases` previously called `generate_from_config(config_path, ...)`
once per case, which re-opened and re-parsed config.yaml from disk on every
iteration. `generate_from_config` now also accepts an already-loaded config
dict, so the YAML file is parsed exactly once regardless of how many cases
are generated. This is a minor win for 3 cases but matters once more cases
are added (Phase 1 is designed to scale to many synthetic targets).
"""

import os

import numpy as np
import pandas as pd
import yaml

from synthetic.noise_models import add_gaussian_noise, add_red_noise, add_stellar_variability
from synthetic.transit_injector import inject_transit, inject_secondary_eclipse


def _load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_time_array(n_points, time_span_days, cadence_minutes, seed=None):
    """
    Returns a 1D numpy array of timestamps in BTJD.

    Starts at 0.0, spaced by cadence_minutes/1440 days.
    Includes small random gaps (~2%) to simulate TESS momentum dumps
    and data downlink interruptions.

    Parameters
    ----------
    n_points : int
        Number of data points before gap removal.
    time_span_days : float
        Total observation window in days.
    cadence_minutes : float
        Sampling interval in minutes.
    seed : int, optional
        Random seed for reproducibility of gap removal.

    Returns
    -------
    np.ndarray
        Time array with ~2% of points removed for gap simulation.
    """
    cadence_days = cadence_minutes / 1440.0
    time = np.arange(0, time_span_days, cadence_days)

    # Truncate to n_points if the arange produced more
    if len(time) > n_points:
        time = time[:n_points]

    # Simulate TESS data gaps: remove ~2% of points randomly
    rng = np.random.default_rng(seed)
    gap_fraction = 0.02
    keep_mask = rng.random(len(time)) > gap_fraction
    time = time[keep_mask]

    return time


def make_base_flux(n_points):
    """
    Returns a 1D numpy array of ones with length n_points.

    This is the flat, noiseless, transitless baseline. All subsequent
    noise and transit injections modify this starting point.

    Parameters
    ----------
    n_points : int
        Length of the flux array.

    Returns
    -------
    np.ndarray
        Array of 1.0 values.
    """
    return np.ones(n_points, dtype=np.float64)


def generate_from_config(config_path, case_name):
    """
    Generates a single synthetic light curve from config.yaml.

    Reads the config, creates time + base flux, applies noise and
    (optionally) injects transit signals.

    Parameters
    ----------
    config_path : str or dict
        Path to config.yaml, OR an already-loaded config dict (avoids
        re-reading/re-parsing the YAML file when generating many cases).
    case_name : str
        Key in config['cases'] (e.g. 'candidate_a').

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, dict)
        (time, flux, metadata_dict)
    """
    config = config_path if isinstance(config_path, dict) else _load_config(config_path)

    gen = config['generation']
    case = config['cases'][case_name]

    seed = case.get('seed', 42)

    # Step 1: Create time array with gaps
    time = make_time_array(
        n_points=gen['n_points'],
        time_span_days=gen['time_span_days'],
        cadence_minutes=gen['cadence_minutes'],
        seed=seed,
    )

    # Step 2: Create flat baseline flux matching time array length
    flux = make_base_flux(len(time))

    # Step 3: Add noise
    noise_type = case.get('noise_type', 'gaussian')
    noise_level = case.get('noise_level', 0.002)

    if noise_type == 'gaussian':
        flux = add_gaussian_noise(flux, sigma=noise_level, seed=seed)
    elif noise_type == 'red':
        flux = add_red_noise(flux, sigma=noise_level, correlation=0.3, seed=seed)
        # For noise-only cases, also add stellar variability
        flux = add_stellar_variability(flux, time, amplitude=0.005,
                                       period_days=12.0, seed=seed)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Step 4: Inject transit (if this case has a transit signal)
    period = case.get('period_days')
    depth = case.get('depth')
    duration = case.get('duration_days')

    if period is not None and depth is not None and duration is not None:
        inject_transit(
            flux, time,
            period_days=period,
            depth=depth,
            duration_days=duration,
            v_shape=case.get('v_shape', False),
        )

        # Step 5: Inject secondary eclipse if configured
        if case.get('secondary_eclipse', False):
            secondary_depth = case.get('secondary_depth', depth * 0.5)
            inject_secondary_eclipse(
                flux, time,
                period_days=period,
                secondary_depth=secondary_depth,
                duration_days=duration,
            )

    # Build metadata dict
    metadata = {
        'cadence_min': gen['cadence_minutes'],
        'time_span_days': gen['time_span_days'],
        'sector': None,  # always None for synthetic
        'label': case.get('label'),
        'true_period': period,
        'true_depth': depth,
        'true_duration': duration,
    }

    return time, flux, metadata


def generate_all_cases(config_path, output_dir):
    """
    Generates all synthetic light curves defined in config.yaml.

    Loops over every case, calls generate_from_config, and writes
    each result as a CSV with 'time' and 'flux' columns.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    output_dir : str
        Directory to write output CSVs. Created if it doesn't exist.
    """
    config = _load_config(config_path)

    os.makedirs(output_dir, exist_ok=True)

    for case_name in config['cases']:
        # Pass the already-loaded config dict so the YAML file is
        # parsed only once for the whole batch, not once per case.
        time, flux, metadata = generate_from_config(config, case_name)

        # Write CSV
        df = pd.DataFrame({'time': time, 'flux': flux})
        csv_path = os.path.join(output_dir, f'{case_name}.csv')
        df.to_csv(csv_path, index=False)

        print(f"  [OK] {case_name}: {len(time)} points, "
              f"label={metadata['label']}, "
              f"flux range=[{flux.min():.4f}, {flux.max():.4f}]")

    print(f"\nAll cases written to {output_dir}/")