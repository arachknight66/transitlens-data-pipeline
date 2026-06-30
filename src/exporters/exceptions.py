"""Exceptions raised while exporting processed datasets."""


class ExportError(RuntimeError):
    """Raised when a processed dataset cannot be exported safely."""
