"""Exceptions raised while reading and parsing FITS light curves."""


class FitsError(RuntimeError):
    """Base exception for FITS processing failures."""


class FitsReadError(FitsError):
    """Raised when a FITS file cannot be opened safely."""


class FitsHduError(FitsError):
    """Raised when no suitable light-curve table HDU exists."""


class FitsColumnError(FitsError):
    """Raised when a required light-curve column is unavailable."""


class FitsValidationError(FitsError):
    """Raised when extracted light-curve arrays are structurally invalid."""


class UnsupportedMissionError(FitsError):
    """Raised when a FITS mission cannot be identified or is unsupported."""
