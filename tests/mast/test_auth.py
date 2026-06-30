"""Tests for MAST client authentication modes."""

from typing import Any

import pytest

from mast.auth import create_mast_client
from mast.exceptions import MastAuthenticationError


class FakeClient:
    """Minimal client used to exercise authentication selection."""

    def query_criteria(self, *args: object, **kwargs: object) -> list[object]:
        """Return no observations."""
        return []

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
        """Return a successful fake download status."""
        return "COMPLETE", None, None


def test_anonymous_client_is_created_without_token() -> None:
    """Anonymous access passes no token to Astroquery."""
    received: list[str | None] = []
    expected = FakeClient()

    def factory(token: str | None) -> FakeClient:
        received.append(token)
        return expected

    assert create_mast_client(client_factory=factory) is expected
    assert received == [None]


def test_api_token_is_passed_to_client_factory() -> None:
    """Token authentication delegates securely to Astroquery."""
    received: list[str | None] = []

    def factory(token: str | None) -> FakeClient:
        received.append(token)
        return FakeClient()

    create_mast_client("secret-token", client_factory=factory)

    assert received == ["secret-token"]


def test_authenticated_client_is_reused() -> None:
    """A caller-managed authenticated session is accepted unchanged."""
    expected = FakeClient()

    assert create_mast_client(authenticated_client=expected) is expected


def test_conflicting_authentication_modes_are_rejected() -> None:
    """A token cannot replace a caller-managed authenticated session."""
    with pytest.raises(MastAuthenticationError, match="either an API token"):
        create_mast_client("secret", FakeClient())


def test_authentication_failure_is_descriptive() -> None:
    """Astroquery authentication errors are converted to domain errors."""

    def failing_factory(token: str | None) -> Any:
        raise OSError("network unavailable")

    with pytest.raises(MastAuthenticationError, match="authentication failed"):
        create_mast_client("bad-token", client_factory=failing_factory)
