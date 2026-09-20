"""Letting the phone find this Mac without being told where it is.

The alternative is typing an IP address into the phone, which works exactly
until the router hands out a different one -- after a reboot, after a day away,
after joining the other band. Then the one thing that is supposed to be a single
tap becomes a debugging session.

So the server announces itself over multicast DNS, the same mechanism behind
every ``something.local`` name on a home network, and the phone looks for the
service rather than the address. Pairing then stores what the machine *is*,
not where it happened to be that afternoon.

The record carries the certificate fingerprint as well, so a phone that already
paired can tell whether the thing answering on this address is the Mac it paired
with, before it sends a token to it.
"""

from __future__ import annotations

import logging
import socket
from typing import Any

logger = logging.getLogger("breeze.discovery")

SERVICE_TYPE = "_breezetts._tcp.local."


class Advertiser:
    """This server's presence on the local network, while it is running."""

    def __init__(self) -> None:
        self._zeroconf: Any = None
        self._info: Any = None

    def start(self, address: str, port: int, properties: dict[str, str]) -> bool:
        """Announce the service. False if zeroconf is not installed."""
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            logger.info(
                "zeroconf is not installed, so the phone will need this Mac's "
                "address typed in: pip install -r requirements.txt"
            )
            return False
        self.stop()
        host = socket.gethostname().split(".")[0]
        try:
            self._zeroconf = Zeroconf()
            self._info = ServiceInfo(
                SERVICE_TYPE,
                f"Breeze TTS on {host}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(address)],
                port=port,
                properties={key: str(value) for key, value in properties.items()},
                server=f"{host}.local.",
            )
            self._zeroconf.register_service(self._info)
        except Exception:  # noqa: BLE001 - an unannounced server still serves
            logger.exception("Could not announce this server on the network")
            self.stop()
            return False
        logger.info("Announced %s at %s:%d", SERVICE_TYPE, address, port)
        return True

    def stop(self) -> None:
        try:
            if self._zeroconf is not None and self._info is not None:
                self._zeroconf.unregister_service(self._info)
        except Exception:  # noqa: BLE001 - shutting down, nothing to salvage
            pass
        finally:
            try:
                if self._zeroconf is not None:
                    self._zeroconf.close()
            except Exception:  # noqa: BLE001
                pass
            self._zeroconf = None
            self._info = None


ADVERTISER = Advertiser()
