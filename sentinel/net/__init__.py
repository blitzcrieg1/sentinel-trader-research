"""Sentinel Trader — network utilities."""

from sentinel.net.dns import GOOGLE_DNS_SERVERS, create_aiohttp_session, install_google_dns_resolver

__all__ = ["GOOGLE_DNS_SERVERS", "create_aiohttp_session", "install_google_dns_resolver"]
