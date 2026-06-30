"""MAST client construction and authentication."""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from astroquery.mast.observations import ObservationsClass
from loguru import logger

from mast.exceptions import MastAuthenticationError


@runtime_checkable
class MastClient(Protocol):
    """Astroquery operations required by this repository."""

    def query_criteria(self, *args: object, **kwargs: object) -> Any:
        """Query MAST observations matching supplied criteria."""

    def get_product_list(self, observations: object) -> Any:
        """Return products associated with one or more observations."""

    def download_file(
        self,
        uri: str,
        *,
        local_path: str,
        cache: bool,
        verbose: bool,
    ) -> tuple[str, str | None, str | None]:
        """Download one MAST data URI to a local path."""


MastClientFactory = Callable[[str | None], MastClient]


def create_mast_client(
    api_token: str | None = None,
    authenticated_client: MastClient | None = None,
    *,
    client_factory: MastClientFactory = ObservationsClass,
) -> MastClient:
    """Create an anonymous, token-authenticated, or injected MAST client.

    Args:
        api_token: Optional MAST API token. It is never logged or persisted.
        authenticated_client: Optional pre-authenticated Astroquery-compatible
            client. This supports callers that manage a session externally.
        client_factory: Injectable Astroquery client constructor.

    Returns:
        A client ready for MAST observation operations.

    Raises:
        MastAuthenticationError: If credentials conflict or token login fails.
    """
    if api_token and authenticated_client is not None:
        message = "provide either an API token or an authenticated client, not both"
        raise MastAuthenticationError(message)

    if authenticated_client is not None:
        logger.info("Using caller-provided authenticated MAST client")
        return authenticated_client

    try:
        client = client_factory(api_token)
    except Exception as error:
        logger.bind(error_type=type(error).__name__).warning(
            "MAST client authentication failed"
        )
        message = "MAST authentication failed"
        raise MastAuthenticationError(message) from error

    access_mode = "token" if api_token else "anonymous"
    logger.bind(access_mode=access_mode).info("MAST client initialized")
    return client
