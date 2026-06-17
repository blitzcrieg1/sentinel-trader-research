"""
DNS workaround for broken Windows/VPN resolvers.

Some environments (NordVPN, router fe80::1) fail to resolve hostnames via
the system DNS.  aiohttp (used by CCXT, Vertex AI, CryptoPanic) inherits
that failure.  Patching ``ClientSession`` to use Google public DNS fixes
MEXC, Vertex, and news fetches without changing system settings.
"""

from __future__ import annotations

import logging
from typing import Final

import aiohttp
from aiohttp.resolver import AsyncResolver

logger = logging.getLogger(__name__)

GOOGLE_DNS_SERVERS: Final[tuple[str, ...]] = ("8.8.8.8", "8.8.4.4")

_installed = False


def create_aiohttp_session(*, timeout_sec: float) -> aiohttp.ClientSession:
    """Build an aiohttp session that resolves hostnames via Google public DNS."""
    resolver = AsyncResolver(nameservers=list(GOOGLE_DNS_SERVERS))
    connector = aiohttp.TCPConnector(resolver=resolver, family=0)
    return aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=timeout_sec),
    )


def install_google_dns_resolver() -> None:
    """Route all new aiohttp sessions through Google DNS. Idempotent."""
    global _installed
    if _installed:
        return

    _orig_init = aiohttp.ClientSession.__init__

    def _init_with_google_dns(self, *args, **kwargs):
        if kwargs.get("connector") is None:
            resolver = AsyncResolver(nameservers=list(GOOGLE_DNS_SERVERS))
            kwargs["connector"] = aiohttp.TCPConnector(resolver=resolver, family=0)
        _orig_init(self, *args, **kwargs)

    aiohttp.ClientSession.__init__ = _init_with_google_dns  # type: ignore[method-assign]
    _installed = True
    logger.info("aiohttp DNS resolver patched -> %s", GOOGLE_DNS_SERVERS)
