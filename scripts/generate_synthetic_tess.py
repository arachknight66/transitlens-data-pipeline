"""Generate reproducible, explicitly synthetic TESS-like light curves."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class Scenario:
    """Truth parameters for one synthetic demonstration light curve."""

    name: str
    classification: str
    period_days: float | None
    primary_depth_ppm: float | None
    duration_hours: float | None
    secondary_depth_ppm: float | None = None


SCENARIOS = (
    Scenario("synthetic-exoplanet", "exoplanet", 3.742, 1850.0, 2.65),
    Scenario(
        "synthetic-eclipsing-binary",
        "eclipsing_binary",
        2.184,
        118000.0,
        4.4,
        54000.0,
    ),
    Scenario("synthetic-variable-star", "other", None, None, None),
)


def _eclipse(
    time: np.ndarray,
    period: float,
    center_phase: float,
    depth: float,
    duration_hours: float,
) -> np.ndarray:
    phase = ((time / period - center_phase + 0.5) % 1.0) - 0.5
    half_width = duration_hours / 24.0 / period / 2.0
    distance = np.abs(phase)
    ingress = max(half_width * 0.18, np.finfo(float).eps)
    profile = np.clip((half_width - distance) / ingress, 0.0, 1.0)
    return depth * profile


def generate(
    scenario: Scenario, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate cadence, flux, and quality arrays for a scenario."""
    rng = np.random.default_rng(seed)
    cadence_days = 10.0 / 60.0 / 24.0
    time = np.arange(0.0, 27.4, cadence_days, dtype=np.float64)
    variability = 0.0012 * np.sin(2.0 * np.pi * time / 6.1 + 0.4)
    variability += 0.00045 * np.sin(2.0 * np.pi * time / 1.37)
    white_noise = rng.normal(0.0, 0.00065, len(time))
    correlated = np.convolve(
        rng.normal(0.0, 0.00022, len(time)), np.ones(9) / 9.0, mode="same"
    )
    relative_flux = 1.0 + variability + white_noise + correlated

    if scenario.classification == "exoplanet":
        relative_flux -= _eclipse(
            time,
            scenario.period_days or 1.0,
            0.17,
            (scenario.primary_depth_ppm or 0.0) / 1_000_000.0,
            scenario.duration_hours or 1.0,
        )
    elif scenario.classification == "eclipsing_binary":
        period = scenario.period_days or 1.0
        relative_flux -= _eclipse(
            time,
            period,
            0.08,
            (scenario.primary_depth_ppm or 0.0) / 1_000_000.0,
            scenario.duration_hours or 1.0,
        )
        relative_flux -= _eclipse(
            time,
            period,
            0.58,
            (scenario.secondary_depth_ppm or 0.0) / 1_000_000.0,
            (scenario.duration_hours or 1.0) * 0.82,
        )
        relative_flux += 0.004 * np.cos(4.0 * np.pi * time / period)
    else:
        relative_flux += 0.0035 * np.sin(2.0 * np.pi * time / 1.83)

    quality = np.zeros(len(time), dtype=np.int32)
    flagged = rng.choice(len(time), size=24, replace=False)
    quality[flagged] = 1
    keep = ~((time > 13.45) & (time < 13.72))
    return time[keep], relative_flux[keep] * 100_000.0, quality[keep]


def write_fits(output: Path, scenario: Scenario, seed: int) -> None:
    """Write one pipeline-compatible synthetic FITS and truth sidecar."""
    time, flux, quality = generate(scenario, seed)
    rng = np.random.default_rng(seed + 10_000)
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "TESS"
    primary.header["MISSION"] = "TESS"
    primary.header["OBJECT"] = scenario.name
    primary.header["TICID"] = f"SYNTH-{seed}"
    primary.header["SYNTH"] = (True, "Synthetic demonstration data")
    primary.header["DATAPROV"] = "TRANSITLENS-SYNTHETIC"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", unit="BJD - 2457000", array=time),
            fits.Column(name="PDCSAP_FLUX", format="E", unit="e-/s", array=flux),
            fits.Column(name="QUALITY", format="J", array=quality),
        ],
        name="LIGHTCURVE",
    )
    table.header["OBJECT"] = scenario.name
    table.header["SYNTH"] = True
    # A realistic auxiliary cutout cube makes the demo exercise substantial
    # FITS transfers while the canonical light-curve table remains unchanged.
    pixels = rng.normal(100.0, 4.5, (len(time), 48, 48)).astype(np.float32)
    axis = np.arange(48, dtype=np.float32) - 23.5
    xx, yy = np.meshgrid(axis, axis)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * 2.2**2)).astype(np.float32)
    pixels += (flux / np.median(flux) * 450.0).astype(np.float32)[:, None, None] * psf
    pixel_hdu = fits.ImageHDU(data=pixels, name="SYNTHPIX")
    pixel_hdu.header["SYNTH"] = True
    pixel_hdu.header["BUNIT"] = "e-/s"
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, table, pixel_hdu]).writeto(
        output, overwrite=True, checksum=True
    )
    output.with_suffix(".truth.json").write_text(
        json.dumps({**asdict(scenario), "synthetic": True, "seed": seed}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate all demonstration fixtures."""
    root = Path(__file__).resolve().parents[2] / "archive" / "synthetic"
    for index, scenario in enumerate(SCENARIOS, start=20260701):
        path = root / f"{scenario.name}.fits"
        write_fits(path, scenario, index)
        print(path)


if __name__ == "__main__":
    main()
