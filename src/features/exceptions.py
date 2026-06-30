"""Exceptions raised during deterministic feature generation."""


class FeatureError(RuntimeError):
    """Raised when a processed light curve cannot produce valid features."""
