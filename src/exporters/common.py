"""Shared deterministic serialization helpers for dataset exporters."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from exporters.exceptions import ExportError


def canonical_json(model: BaseModel) -> bytes:
    """Serialize a Pydantic model as stable UTF-8 JSON bytes.

    Args:
        model: Validated model to serialize.

    Returns:
        Canonical compact JSON bytes with sorted keys.
    """
    value: Any = model.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def validate_output_path(path: Path, suffix: str) -> Path:
    """Validate and prepare a caller-selected export destination.

    Args:
        path: Requested output file.
        suffix: Required lowercase format suffix.

    Returns:
        Resolved output path with an existing parent directory.

    Raises:
        ExportError: If the filename has the wrong extension.
    """
    if path.suffix.casefold() != suffix:
        raise ExportError(f"export path must use the '{suffix}' extension")
    destination = path.expanduser().resolve()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        message = f"could not prepare export directory: {destination.parent}"
        raise ExportError(message) from error
    return destination
