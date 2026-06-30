"""Deterministic MAST observation search."""

from collections.abc import Mapping
from typing import Any

from loguru import logger

from mast.auth import MastClient
from mast.exceptions import MastSearchError
from mast.models import Mission, Observation, ObservationSearch


def search_observations(
    criteria: ObservationSearch,
    client: MastClient,
) -> list[Observation]:
    """Search MAST for supported light-curve observations.

    Args:
        criteria: Validated target, mission, radius, and result limit.
        client: Configured MAST client.

    Returns:
        Deterministically ordered observation records.

    Raises:
        MastSearchError: If MAST is unavailable or returns malformed records.
    """
    logger.bind(
        target=criteria.target,
        missions=[mission.value for mission in criteria.missions],
    ).info("Searching MAST observations")
    try:
        result = client.query_criteria(
            objectname=criteria.target,
            radius=f"{criteria.radius_deg} deg",
            obs_collection=[mission.value for mission in criteria.missions],
            dataproduct_type="timeseries",
        )
        observations = [_parse_observation(row) for row in result]
    except Exception as error:
        logger.bind(error_type=type(error).__name__).warning(
            "MAST observation search failed"
        )
        message = f"MAST observation search failed for target '{criteria.target}'"
        raise MastSearchError(message) from error

    observations.sort(
        key=lambda item: (item.mission.value, item.observation_id, item.mast_id)
    )
    limited = observations[: criteria.limit]
    logger.bind(result_count=len(limited)).info("MAST observation search completed")
    return limited


def _parse_observation(row: object) -> Observation:
    """Convert one Astroquery result row to the public observation model."""
    mission = Mission(str(_required_value(row, "obs_collection")))
    mast_id = str(_required_value(row, "obsid"))
    observation_id = str(_value(row, "obs_id") or mast_id)
    return Observation(
        mast_id=mast_id,
        observation_id=observation_id,
        target_name=str(_required_value(row, "target_name")),
        mission=mission,
        product_type=str(_required_value(row, "dataproduct_type")),
        start_time=_optional_float(_value(row, "t_min")),
        end_time=_optional_float(_value(row, "t_max")),
    )


def _required_value(row: object, name: str) -> Any:
    """Read a required result value or reject a malformed response."""
    value = _value(row, name)
    if value is None or str(value).strip() == "":
        message = f"MAST response is missing required field '{name}'"
        raise ValueError(message)
    return value


def _value(row: object, name: str) -> Any | None:
    """Read a value from an Astropy row or mapping."""
    column_names = getattr(row, "colnames", ())
    if name in column_names:
        value = row[name]  # type: ignore[index]
    elif isinstance(row, Mapping):
        value = row.get(name)
    else:
        return None
    return None if bool(getattr(value, "mask", False)) else value


def _optional_float(value: object | None) -> float | None:
    """Convert an optional table value to a native float."""
    return None if value is None else float(value)
