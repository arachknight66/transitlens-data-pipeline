"""Exceptions raised by deterministic light-curve preprocessing."""


class PreprocessingError(RuntimeError):
    """Base exception for preprocessing failures."""


class InvalidMeasurementsError(PreprocessingError):
    """Raised when cleaning leaves no scientifically usable measurements."""


class NormalizationError(PreprocessingError):
    """Raised when flux cannot be normalized safely."""


class FilteringError(PreprocessingError):
    """Raised when a filtering configuration or signal is invalid."""
