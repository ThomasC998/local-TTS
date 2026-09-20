#!/usr/bin/env python3
"""What guards the server once it is on the Wi-Fi rather than on loopback.

The checks here are the ones whose failure would be silent. A token that is
world-readable, a certificate that does not name the address the phone dials, a
guard that lets a query parameter through on a route that was meant to require a
header -- none of those break anything visibly. They just quietly mean the
protection is not there.

    python test_mobile_auth.py

Nothing listens on a port and nothing is written outside a temporary directory.
"""

from __future__ import annotations

import ipaddress
import os
import stat
import sys
import tempfile
from pathlib import Path

_TEMP = tempfile.mkdtemp(prefix="breeze-auth-test-")
os.environ["BREEZE_STATE_DIR"] = str(Path(_TEMP) / "state")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import mobile_auth  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  \033[32m✓\033[0m {label}")
    else:
        FAILED += 1
        print(f"  \033[31m✗\033[0m {label}" + (f" -- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------------
section("The token")
# ---------------------------------------------------------------------------
token = mobile_auth.load_token()
check("a token is made on first use", bool(token) and len(token) >= 40, str(token))
check("and it is the same one next time", mobile_auth.load_token() == token)

mode = stat.S_IMODE(mobile_auth.TOKEN_PATH.stat().st_mode)
check("the file is readable by nobody else", mode == 0o600, oct(mode))
check("the token is not in the filename or the directory listing",
      token not in str(mobile_auth.TOKEN_PATH))

rotated = mobile_auth.rotate_token()
check("rotating makes a different one", rotated != token)
check("...and that is what is loaded afterwards",
      mobile_auth.load_token() == rotated)
check("the rotated file is still private",
      stat.S_IMODE(mobile_auth.TOKEN_PATH.stat().st_mode) == 0o600)


# ---------------------------------------------------------------------------
section("The certificate")
# ---------------------------------------------------------------------------
certificate, key = mobile_auth.ensure_certificate()
check("a certificate and a key are written",
      certificate.is_file() and key.is_file())
check("the private key is readable by nobody else",
      stat.S_IMODE(key.stat().st_mode) == 0o600,
      oct(stat.S_IMODE(key.stat().st_mode)))

fingerprint = mobile_auth.fingerprint()
check("it has a SHA-256 fingerprint to pin",
      bool(fingerprint) and len(fingerprint) == 64, str(fingerprint))

again, _ = mobile_auth.ensure_certificate()
check("asking twice does not replace it",
      mobile_auth.fingerprint() == fingerprint)

from cryptography import x509  # noqa: E402

parsed = x509.load_pem_x509_certificate(certificate.read_bytes())
names = parsed.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
addresses = names.get_values_for_type(x509.IPAddress)
hostnames = names.get_values_for_type(x509.DNSName)
check("loopback is covered, so the web UI works over TLS too",
      ipaddress.ip_address("127.0.0.1") in addresses, str(addresses))
check("the .local name is covered", any(n.endswith(".local") for n in hostnames),
      str(hostnames))
lan = mobile_auth.lan_address()
if lan:
    check("this Mac's network address is covered, which is what the phone dials",
          ipaddress.ip_address(lan) in addresses, f"{lan} not in {addresses}")
else:
    check("no network address to cover right now (skipped)", True)


# ---------------------------------------------------------------------------
section("The guard")
# ---------------------------------------------------------------------------
app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/voices")
def voices() -> dict:
    return {"voices": []}


@app.get("/v1/read/rd_x/p0.wav")
def paragraph() -> dict:
    return {"audio": True}


app.add_middleware(mobile_auth.TokenAuthMiddleware, token=rotated)
client = TestClient(app)

check("health answers without a token, so the phone can find the Mac",
      client.get("/health").status_code == 200)
check("everything else refuses one", client.get("/v1/voices").status_code == 401)

# The web UI and the hotkey daemon run on this machine and are not asked for a
# token -- but "on this machine" has to mean both ends of it. A page served
# from a name that resolves to 127.0.0.1 makes requests from a local socket
# too, so the Host header is what tells a rebinding attempt from the real UI.
local = TestClient(app, client=("127.0.0.1", 54321))
check("this machine's own web UI needs no token",
      local.get("/v1/voices", headers={"Host": "127.0.0.1:7860"}).status_code == 200)
check("...nor under its other name",
      local.get("/v1/voices", headers={"Host": "localhost:7860"}).status_code == 200)
check("but a local socket carrying somebody else's Host is challenged",
      local.get("/v1/voices", headers={"Host": "evil.example.com"}).status_code == 401)
check("...and says how to authenticate",
      client.get("/v1/voices").headers.get("www-authenticate") == "Bearer")
check("a wrong token is refused",
      client.get("/v1/voices",
                 headers={"Authorization": "Bearer nope"}).status_code == 401)
check("the right token is accepted",
      client.get("/v1/voices",
                 headers={"Authorization": f"Bearer {rotated}"}).status_code == 200)
check("a token in the query string is refused on an ordinary route",
      client.get(f"/v1/voices?t={rotated}").status_code == 401)
check("but accepted on audio, which a media player fetches without headers",
      client.get(f"/v1/read/rd_x/p0.wav?t={rotated}").status_code == 200)
check("and a wrong one there is still refused",
      client.get("/v1/read/rd_x/p0.wav?t=nope").status_code == 401)


# ---------------------------------------------------------------------------
section("Guessing at it")
# ---------------------------------------------------------------------------
locked = FastAPI()


@locked.get("/v1/voices")
def locked_voices() -> dict:
    return {"voices": []}


locked.add_middleware(mobile_auth.TokenAuthMiddleware, token=rotated)
attacker = TestClient(locked)

codes = [
    attacker.get("/v1/voices", headers={"Authorization": "Bearer wrong"}).status_code
    for _ in range(mobile_auth.GUESS_LIMIT + 2)
]
check("the first wrong guesses are refused",
      codes[0] == 401 and codes[1] == 401, str(codes[:3]))
check("after too many, the address is shut out entirely",
      codes[-1] == 429, str(codes))
check("and the right token does not help while it is shut out",
      attacker.get("/v1/voices",
                   headers={"Authorization": f"Bearer {rotated}"}).status_code == 429)


# ---------------------------------------------------------------------------
section("Pairing")
# ---------------------------------------------------------------------------
payload = mobile_auth.pairing_payload(7860, {"mac_addresses": ["aa:bb:cc:dd:ee:ff"]})
check("the payload carries the token", payload["token"] == rotated)
check("...the fingerprint to pin", payload["fingerprint"] == fingerprint)
check("...the port", payload["port"] == 7860)
check("...and what was passed in", payload["mac_addresses"] == ["aa:bb:cc:dd:ee:ff"])

svg = mobile_auth.pairing_qr_svg(payload)
check("it renders as an inline SVG, needing no image library",
      svg is not None and svg.lstrip().startswith("<?xml"), str(svg)[:60])
check("the QR is the only thing carrying the token to the phone",
      svg is not None and rotated not in svg)


# ---------------------------------------------------------------------------
print(f"\n{PASSED} passed, {FAILED} failed\n")
sys.exit(1 if FAILED else 0)
