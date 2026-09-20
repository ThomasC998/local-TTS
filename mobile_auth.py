"""What has to be true before this server is reachable from anything but this Mac.

Until now the server bound ``127.0.0.1`` and had no authentication, which was
exactly right: the only thing that could reach it was this machine, and the only
person on this machine is its owner. Putting it on the Wi-Fi changes that
completely. Every route here can do something worth stealing -- ``/v1/speak``
plays audio out loud, ``/v1/speak/clipboard`` hands over whatever was last
copied, the voice routes hold recordings of a real person's voice -- so the
network has to be treated as hostile before the first byte is served to it.

Three things, and none of them optional once the bind is not loopback:

*A token.* Thirty-two random bytes, generated once, stored `0600`, compared with
``hmac.compare_digest`` and never logged. Every request carries it, as a bearer
header or -- for audio URLs, which media players fetch without headers of their
own -- as a query parameter.

*A limit on guessing.* Ten wrong tokens from one address in a minute and that
address gets ``429`` for a while. A token this long is not guessable in any
practical sense, but a lockout costs nothing and turns "not practical" into
"not possible", and the log line is how you would ever know somebody tried.

*A certificate.* Self-signed, generated here, with the LAN address and the
``.local`` name in it. On its own a self-signed certificate proves nothing --
but the phone is handed the certificate's fingerprint at pairing, in a QR code,
over a channel that is a photograph rather than a network. After that the phone
refuses anything that is not exactly this certificate, so nothing on the shared
Wi-Fi can read or change a single request, which is the whole point.

The pairing payload is deliberately small and is only ever served to loopback:
the QR is meant to be looked at on this screen, not fetched over the network it
is granting access to.
"""

from __future__ import annotations

import base64
import datetime
import hmac
import ipaddress
import json
import logging
import os
import secrets
import socket
import threading
import time
from pathlib import Path
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from archive import STATE_DIR

logger = logging.getLogger("breeze.auth")

TOKEN_PATH = STATE_DIR / "mobile_token.json"
TLS_DIR = STATE_DIR / "tls"
CERT_PATH = TLS_DIR / "server.crt"
KEY_PATH = TLS_DIR / "server.key"

# Routes that answer without a token. ``/health`` is how the phone finds out
# whether the Mac is awake yet, which it has to be able to ask before it can be
# trusted with anything -- and the answer says nothing a scanner does not know.
OPEN_PATHS = frozenset({"/health"})

# Addresses that are this machine talking to itself.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

# What a request from this Mac is called. The hotkeys, the web UI and anything
# else running here share it: they are all the same pair of speakers.
LOCAL_ORIGIN = {"kind": "local", "id": "mac", "name": "this Mac"}

# Routes whose token may ride in the query string. Audio is fetched by the
# media player rather than by our own HTTP client, and a player does not attach
# headers to the URLs in a playlist.
QUERY_TOKEN_PREFIXES = ("/v1/read/",)

# Wrong tokens from one address before it is shut out, and for how long.
GUESS_LIMIT = 10
GUESS_WINDOW_SECONDS = 60.0
LOCKOUT_SECONDS = 300.0

# The certificate is for a machine on a home network, not a public site. Long
# enough not to be a chore, short enough to be worth re-pairing eventually.
CERT_DAYS = 825


class AuthUnavailable(RuntimeError):
    """The server cannot be exposed safely, with the reason attached."""


# ---------------------------------------------------------------------------
# The devices, and their tokens
#
# One token per paired device rather than one for the installation. It costs
# nothing and buys two things worth having: the server knows *which* device is
# asking without being told -- the proof is which key verified, which cannot be
# claimed falsely the way a name in a request can -- and a phone that is lost
# can be revoked without re-pairing everything else.
# ---------------------------------------------------------------------------
def _write_private(path: Path, data: str) -> None:
    """Write a secret so that only this user can read it, without a window.

    Created with the right mode from the start rather than chmod-ed afterwards:
    between the two there is a moment where the file exists and is readable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(data)
    os.replace(tmp, path)


def _read_store() -> list[dict[str, Any]]:
    """Every paired device. An older single-token file is migrated on sight."""
    try:
        stored = json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception:  # noqa: BLE001 - a corrupt store is replaced, not fatal
        logger.warning("The paired-device file could not be read; starting fresh")
        return []
    if isinstance(stored, dict) and stored.get("token"):
        return [
            {
                "id": "device-1",
                "name": "phone",
                "token": str(stored["token"]),
                "created": stored.get("created") or time.time(),
            }
        ]
    devices = stored.get("devices") if isinstance(stored, dict) else None
    return [device for device in (devices or []) if device.get("token")]


def _write_store(devices: list[dict[str, Any]]) -> None:
    _write_private(TOKEN_PATH, json.dumps({"devices": devices}, indent=2))


def devices() -> list[dict[str, Any]]:
    """The paired devices, without their tokens -- for showing on a page."""
    return [
        {key: device[key] for key in ("id", "name", "created") if key in device}
        for device in _read_store()
    ]


def verify(token: str) -> dict[str, Any] | None:
    """Which device this token belongs to, if any.

    Every stored token is compared, and always all of them: returning early on
    the first match would make the time taken depend on which device asked,
    which is a small thing to leak but a free one not to.
    """
    found: dict[str, Any] | None = None
    for device in _read_store():
        if hmac.compare_digest(token, str(device.get("token") or "")):
            found = device
    return found


def device_for(name: str = "phone", create: bool = True) -> dict[str, Any] | None:
    """The device paired under this name, pairing one if there is none."""
    for device in _read_store():
        if device.get("name") == name:
            return device
    return issue_device(name) if create else None


def issue_device(name: str = "phone") -> dict[str, Any]:
    """Pair another device, with a token of its own."""
    existing = _read_store()
    device = {
        "id": f"dev_{secrets.token_hex(6)}",
        "name": name.strip() or "phone",
        "token": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("="),
        "created": time.time(),
    }
    existing.append(device)
    _write_store(existing)
    logger.info("Paired a new device (%s)", device["name"])
    return device


def revoke_device(device_id: str) -> bool:
    """Un-pair one device. The others carry on working."""
    existing = _read_store()
    remaining = [device for device in existing if device.get("id") != device_id]
    if len(remaining) == len(existing):
        return False
    _write_store(remaining)
    logger.info("Revoked device %s", device_id)
    return True


def load_token(create: bool = True) -> str | None:
    """The default device's token. The single-device shorthand."""
    device = device_for("phone", create=create)
    return str(device["token"]) if device else None


def rotate_token() -> str:
    """Un-pair everything and start again with one fresh device."""
    _write_store([])
    return str(issue_device("phone")["token"])


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------
def lan_address() -> str | None:
    """This Mac's address on the network it would be reached over.

    Asked by opening a UDP socket towards a public address and reading back
    which local interface the kernel chose. Nothing is sent, and it works
    without a route to that address actually existing.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1: reserved, never routed
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return None if address.startswith("127.") else address


def fingerprint(cert_path: Path = CERT_PATH) -> str | None:
    """The certificate's SHA-256, formatted the way the phone will compare it."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        return None
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    except Exception:  # noqa: BLE001 - no certificate yet is not an error here
        return None
    return cert.fingerprint(hashes.SHA256()).hex()


def ensure_certificate(force: bool = False) -> tuple[Path, Path]:
    """The TLS certificate and key, made on first use.

    Regenerated when the LAN address it was made for is no longer this machine's
    -- a certificate that does not name the address the phone dials is a
    certificate the phone will reject, and re-pairing is cheaper than debugging
    that.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise AuthUnavailable(
            "TLS needs the 'cryptography' package: pip install -r requirements.txt"
        ) from exc

    address = lan_address()
    if not force and CERT_PATH.exists() and KEY_PATH.exists():
        if address is None or _certificate_covers(address):
            return CERT_PATH, KEY_PATH
        logger.info("This Mac's address changed; making a new certificate")

    host = socket.gethostname().split(".")[0]
    names: list[Any] = [x509.DNSName(f"{host}.local"), x509.DNSName("localhost")]
    if address:
        names.append(x509.IPAddress(ipaddress.ip_address(address)))
    names.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, f"Breeze TTS on {host}")]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CERT_DAYS))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .sign(key, hashes.SHA256())
    )

    TLS_DIR.mkdir(parents=True, exist_ok=True)
    _write_private(
        KEY_PATH,
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode(),
    )
    CERT_PATH.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    logger.info("Wrote a TLS certificate for %s to %s", address or host, CERT_PATH)
    return CERT_PATH, KEY_PATH


def _certificate_covers(address: str) -> bool:
    try:
        from cryptography import x509
    except ImportError:
        return True
    try:
        cert = x509.load_pem_x509_certificate(CERT_PATH.read_bytes())
        alt = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        return ipaddress.ip_address(address) in alt.get_values_for_type(x509.IPAddress)
    except Exception:  # noqa: BLE001 - an unreadable certificate is replaced
        return False


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
class _Guesses:
    """Wrong tokens, per address, with a lockout once there are too many."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: dict[str, list[float]] = {}
        self._locked: dict[str, float] = {}

    def locked(self, address: str) -> bool:
        with self._lock:
            until = self._locked.get(address, 0.0)
            if until > time.monotonic():
                return True
            self._locked.pop(address, None)
            return False

    def wrong(self, address: str) -> None:
        now = time.monotonic()
        with self._lock:
            recent = [
                stamp
                for stamp in self._attempts.get(address, [])
                if now - stamp < GUESS_WINDOW_SECONDS
            ]
            recent.append(now)
            self._attempts[address] = recent
            if len(recent) >= GUESS_LIMIT:
                self._locked[address] = now + LOCKOUT_SECONDS
                self._attempts.pop(address, None)
                logger.warning(
                    "Too many bad tokens from %s; shutting it out for %.0f minutes",
                    address,
                    LOCKOUT_SECONDS / 60,
                )

    def right(self, address: str) -> None:
        with self._lock:
            self._attempts.pop(address, None)


def _is_loopback(request: Any) -> bool:
    """Whether this request is the Mac talking to itself, by both measures.

    The client address alone is not enough. In a DNS-rebinding attack the
    connection genuinely does come from 127.0.0.1 -- it is the browser on this
    machine making it -- while the ``Host`` header carries the attacker's name.
    Requiring both to be loopback is what tells the two apart.
    """
    client = request.client.host if request.client else ""
    if client not in LOOPBACK:
        return False
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
    return host in LOOPBACK


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Requires the pairing token on everything except ``/health``.

    Installed only when the server is bound somewhere other than loopback. On
    loopback it would be pure friction: anything that can reach the socket can
    already read the token file it would be checking against.
    """

    def __init__(self, app: Any, token: str | None = None) -> None:
        super().__init__(app)
        self._guesses = _Guesses()

    async def dispatch(self, request: Any, call_next: Any) -> Any:
        path = request.url.path
        if path in OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        address = request.client.host if request.client else "unknown"
        if _is_loopback(request):
            # This machine talking to itself: the web UI, the hotkey daemon, the
            # pairing page. A token here would guard nothing -- whatever is
            # running as this user can read the token file it would be checked
            # against. What is checked is that the request really is local,
            # ``Host`` included, so a web page cannot point a name at 127.0.0.1
            # and have the browser make the call on its behalf.
            request.state.origin = LOCAL_ORIGIN
            return await call_next(request)
        if self._guesses.locked(address):
            return JSONResponse({"detail": "Too many attempts"}, status_code=429)

        offered = self._offered(request, path)
        device = verify(offered) if offered else None
        if device is None:
            self._guesses.wrong(address)
            return JSONResponse(
                {"detail": "This server is paired to a device; token required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        self._guesses.right(address)
        # Which device asked is decided here and nowhere else. It is the key
        # that verified, not anything the request said about itself, so a
        # device cannot ask as another one by claiming to be it.
        request.state.origin = {
            "kind": "device",
            "id": device.get("id", "device"),
            "name": device.get("name", "phone"),
        }
        return await call_next(request)

    def _offered(self, request: Any, path: str) -> str | None:
        header = request.headers.get("authorization") or ""
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        if path.startswith(QUERY_TOKEN_PREFIXES):
            return request.query_params.get("t")
        return None


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def pairing_payload(
    port: int, extra: dict[str, Any] | None = None, device_name: str = "phone"
) -> dict[str, Any]:
    """Everything one device needs to talk to this Mac and nothing more."""
    device = device_for(device_name) or {}
    payload = {
        "v": 1,
        "host": lan_address(),
        "name": socket.gethostname().split(".")[0],
        "port": port,
        "token": device.get("token"),
        "device_id": device.get("id"),
        "fingerprint": fingerprint(),
    }
    payload.update(extra or {})
    return payload


def pairing_qr_svg(payload: dict[str, Any]) -> str | None:
    """The payload as an inline SVG QR code, or None if ``qrcode`` is missing.

    SVG rather than a PNG so the settings page needs no image encoding, no
    Pillow, and no file on disk holding the token as a picture.
    """
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return None
    text = json.dumps(payload, separators=(",", ":"))
    image = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, border=2)
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer)
    return buffer.getvalue().decode("utf-8")
