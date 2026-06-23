"""
tests/test_noise_models.py
─────────────────────────────
Unit tests for synthetic/noise_models.py (Phase 1C).
"""

import numpy as np

from synthetic.noise_models import (
    add_gaussian_noise,
    add_red_noise,
    add_stellar_variability,
)


def test_gaussian_noise_mean(synthetic_flux):
    # after adding Gaussian noise, mean should still be ~1.0
    flux = np.array(synthetic_flux)
    noisy = add_gaussian_noise(flux, sigma=0.01, seed=1)
    assert abs(noisy.mean() - 1.0) < 0.01


def test_gaussian_noise_std(synthetic_flux):
    # std of (noisy - original) should be close to sigma parameter
    flux = np.ones(5000)
    sigma = 0.01
    noisy = add_gaussian_noise(flux, sigma=sigma, seed=2)
    residual_std = (noisy - flux).std()
    assert abs(residual_std - sigma) < sigma * 0.15


def test_red_noise_correlated():
    # consecutive residuals should have positive autocorrelation at lag 1
    flux = np.ones(5000)
    noisy = add_red_noise(flux, sigma=0.01, correlation=0.6, seed=3)
    residual = noisy - flux
    lag1_corr = np.corrcoef(residual[:-1], residual[1:])[0, 1]
    assert lag1_corr > 0.2


def test_seed_reproducibility(synthetic_flux):
    # same seed -> identical output; different seed -> different output
    flux = np.array(synthetic_flux)

    a = add_gaussian_noise(flux, sigma=0.01, seed=42)
    b = add_gaussian_noise(flux, sigma=0.01, seed=42)
    c = add_gaussian_noise(flux, sigma=0.01, seed=43)

    assert np.allclose(a, b)
    assert not np.allclose(a, c)

    # same property should hold for red noise and stellar variability
    time = np.linspace(0, 5, len(flux))

    red_a = add_red_noise(flux, sigma=0.01, correlation=0.3, seed=7)
    red_b = add_red_noise(flux, sigma=0.01, correlation=0.3, seed=7)
    red_c = add_red_noise(flux, sigma=0.01, correlation=0.3, seed=8)
    assert np.allclose(red_a, red_b)
    assert not np.allclose(red_a, red_c)

    var_a = add_stellar_variability(flux, time, amplitude=0.005, period_days=12.0, seed=9)
    var_b = add_stellar_variability(flux, time, amplitude=0.005, period_days=12.0, seed=9)
    var_c = add_stellar_variability(flux, time, amplitude=0.005, period_days=12.0, seed=10)
    assert np.allclose(var_a, var_b)
    assert not np.allclose(var_a, var_c)