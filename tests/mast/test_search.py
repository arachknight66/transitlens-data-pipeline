"""Tests for deterministic MAST observation search."""

from pathlib import Path

import pytest
from astropy.table import Table
from pydantic import ValidationError

from mast.exceptions import MastSearchError
from mast.models import Mission, ObservationSearch
from mast.search import search_observations


class SearchClient:
    """Configurable fake MAST search client."""

    def __init__(self, rows: object) -> None:
        """Store the search result or exception."""
        self.rows = rows
        self.criteria: dict[str, object] = {}

    def query_criteria(self, *args: object, **kwargs: object) -> object:
        """Return configured rows and retain submitted criteria."""
        self.criteria = kwargs
        if isinstance(self.rows, Exception):
            raise self.rows
        return self.rows

    def get_product_list(self, observations: object) -> list[object]:
        """Return no products."""
        return []

    def download_file(
        self,
        uri: str,
        *,
        local_path: str,
        cache: bool,
        verbose: bool,
    ) -> tuple[str, str | None, str | None]:
        """Return a successful unused status."""
        Path(local_path).write_bytes(b"unused")
        return "COMPLETE", None, None


def _observation_table() -> Table:
    """Create intentionally unordered representative MAST results."""
    return Table(
        rows=[
            ("20", "tess-b", "Beta", "TESS", "timeseries", 2.0, 3.0),
            ("10", "kepler-a", "Alpha", "Kepler", "timeseries", 1.0, 2.0),
            ("11", "k2-a", "Alpha", "K2", "timeseries", None, None),
        ],
        names=(
            "obsid",
            "obs_id",
            "target_name",
            "obs_collection",
            "dataproduct_type",
            "t_min",
            "t_max",
        ),
    )


def test_search_submits_supported_criteria_and_sorts_results() -> None:
    """Search results are normalized and ordered independent of MAST order."""
    client = SearchClient(_observation_table())
    criteria = ObservationSearch(target="  TIC 123  ", limit=2)

    results = search_observations(criteria, client)

    assert [result.observation_id for result in results] == ["k2-a", "kepler-a"]
    assert client.criteria == {
        "objectname": "TIC 123",
        "radius": "0.001 deg",
        "obs_collection": ["Kepler", "K2", "TESS"],
        "dataproduct_type": "timeseries",
    }
    assert results[0].start_time is None


def test_search_supports_mapping_rows_and_obsid_fallback() -> None:
    """Mapping responses without obs_id use the stable numeric MAST ID."""
    client = SearchClient(
        [
            {
                "obsid": 42,
                "target_name": "Target",
                "obs_collection": "TESS",
                "dataproduct_type": "timeseries",
            }
        ]
    )

    result = search_observations(ObservationSearch(target="Target"), client)

    assert result[0].mast_id == "42"
    assert result[0].observation_id == "42"


def test_search_failure_is_wrapped_with_target_context() -> None:
    """Network failures become descriptive search exceptions."""
    client = SearchClient(ConnectionError("offline"))

    with pytest.raises(MastSearchError, match="Target Name"):
        search_observations(ObservationSearch(target="Target Name"), client)


def test_malformed_search_response_is_rejected() -> None:
    """Missing required MAST columns do not produce partial records."""
    client = SearchClient([{"obsid": "1", "obs_collection": "TESS"}])

    with pytest.raises(MastSearchError) as error:
        search_observations(ObservationSearch(target="Target"), client)

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("target", ["", "   "])
def test_empty_target_is_rejected(target: str) -> None:
    """Search requires a meaningful target name."""
    with pytest.raises(ValidationError, match="must not be empty"):
        ObservationSearch(target=target)


def test_missions_are_required_and_deduplicated() -> None:
    """Mission criteria remain non-empty and deterministic."""
    criteria = ObservationSearch(
        target="Target", missions=(Mission.TESS, Mission.TESS, Mission.K2)
    )
    assert criteria.missions == (Mission.TESS, Mission.K2)

    with pytest.raises(ValidationError, match="at least one mission"):
        ObservationSearch(target="Target", missions=())
