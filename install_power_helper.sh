#!/bin/bash
#
# Let this user run exactly two pmset commands without a password, so the
# server can hold the Mac awake while the phone is listening and let it sleep
# afterwards.
#
#     sudo ./install_power_helper.sh            install
#     sudo ./install_power_helper.sh --remove   undo it
#
# Why this is needed at all: closing a MacBook's lid with no external display
# triggers clamshell sleep, which sits below every user-space power assertion --
# caffeinate, Amphetamine, and anything this server could do on its own are all
# ignored by it. The only switch that stops it is `pmset disablesleep`, which is
# a system-wide setting and therefore needs root.
#
# What is actually granted: two exact command lines, nothing else. Not "pmset",
# not "pmset anything" -- these three, matched in full by sudo:
#
#     /usr/bin/pmset -a disablesleep 1      stop sleeping (lid closed)
#     /usr/bin/pmset -a disablesleep 0      sleep normally again
#     /usr/bin/pmset sleepnow               sleep now
#
# Removing it is one command and is printed at the end. Nothing else on the
# machine changes.

set -euo pipefail

RULE_FILE="/etc/sudoers.d/breezetts-power"
TARGET_USER="${SUDO_USER:-$(id -un)}"

if [[ $EUID -ne 0 ]]; then
    echo "This has to run as root:  sudo $0" >&2
    exit 1
fi

if [[ "${1:-}" == "--remove" ]]; then
    if [[ -f "$RULE_FILE" ]]; then
        rm -f "$RULE_FILE"
        echo "Removed $RULE_FILE. The server can no longer change sleep settings."
    else
        echo "Nothing to remove; $RULE_FILE does not exist."
    fi
    exit 0
fi

if [[ "$TARGET_USER" == "root" ]]; then
    echo "Run this with sudo from your own account, not as root directly." >&2
    exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

cat >"$TMP" <<RULES
# Installed by Breeze TTS 2 (install_power_helper.sh).
# Lets the server hold this Mac awake for a read from the phone, and let it
# sleep again afterwards. Three exact commands, no arguments of their choosing.
Cmnd_Alias BREEZE_POWER = /usr/bin/pmset -a disablesleep 1, \\
                          /usr/bin/pmset -a disablesleep 0, \\
                          /usr/bin/pmset sleepnow
$TARGET_USER ALL=(root) NOPASSWD: BREEZE_POWER
RULES

# A syntactically invalid file in sudoers.d can lock you out of sudo entirely,
# so it is checked before it is ever put in place.
if ! visudo -c -f "$TMP" >/dev/null; then
    echo "Refusing to install: the generated rule did not pass visudo." >&2
    exit 1
fi

install -m 0440 -o root -g wheel "$TMP" "$RULE_FILE"

echo "Installed $RULE_FILE for user '$TARGET_USER'."
echo
echo "Granted, and nothing else:"
echo "    pmset -a disablesleep 1 | 0     hold the Mac awake with the lid shut"
echo "    pmset sleepnow                  send it to sleep"
echo
echo "To undo:   sudo $0 --remove"
echo
echo "One more setting, in System Settings > Battery > Options:"
echo "    'Wake for network access' must be on for the phone to wake this Mac."
echo "    Current value: $(/usr/bin/pmset -g | awk '/womp/ {print $2}')  (1 = on)"
