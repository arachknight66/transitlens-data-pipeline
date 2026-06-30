"""Exceptions raised by the MAST integration layer."""


class MastError(RuntimeError):
    """Base exception for MAST integration failures."""


class MastAuthenticationError(MastError):
    """Raised when MAST authentication cannot be completed."""


class MastSearchError(MastError):
    """Raised when an observation search fails."""


class MastDownloadError(MastError):
    """Raised when a MAST product cannot be downloaded safely."""


class MastProductNotFoundError(MastDownloadError):
    """Raised when an observation has no supported light-curve FITS product."""
