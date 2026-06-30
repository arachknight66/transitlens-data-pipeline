"""Injectable service dependencies for REST route handlers."""

from collections.abc import Callable
from dataclasses import dataclass

from config import Settings
from mast.auth import MastClient, create_mast_client


@dataclass(frozen=True)
class ApiServices:
    """Factories for external services used by API requests."""

    mast_client_provider: Callable[[], MastClient]

    @classmethod
    def from_settings(cls, settings: Settings) -> "ApiServices":
        """Create production service providers from validated settings.

        Args:
            settings: Runtime application settings.

        Returns:
            Immutable service provider container.
        """
        return cls(
            mast_client_provider=lambda: create_mast_client(settings.mast_api_token)
        )
