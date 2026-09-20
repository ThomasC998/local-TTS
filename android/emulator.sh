#!/bin/bash
#
# The simulated phone: install it, or take it off this Mac again.
#
#     ./emulator.sh status      what is here and what it costs
#     ./emulator.sh install     put it all back (~5 GB download)
#     ./emulator.sh remove      take it off, keep adb for real phones
#     ./emulator.sh remove --all  take adb and the tools off too
#
# Why this is a script and not a paragraph in a README: it is six or seven
# steps, it is done rarely, and the one thing worse than not having the
# emulator is having half of it. Both directions are here so that removing it
# is not a decision you have to be brave about -- `install` puts back exactly
# what `remove` took.
#
# Nothing in this project needs the emulator to build or ship. The APK is built
# in CI and the JVM tests in android/app/src/test/ run in seconds without a
# device. This is only for watching the app actually play something.

set -euo pipefail

BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
export ANDROID_HOME="${ANDROID_HOME:-$BREW_PREFIX/share/android-commandlinetools}"
export JAVA_HOME="${JAVA_HOME:-$BREW_PREFIX/opt/openjdk@17}"

AVD="${AVD:-phone}"
IMAGE="${IMAGE:-system-images;android-35;google_apis;arm64-v8a}"
DEVICE="${DEVICE:-pixel_6}"

# The AVD asks for 12 GB of userdata by default and will not start without it.
# The partition is sparse -- this is a ceiling, not an allocation -- so the
# number only has to be larger than the app and its data will ever be.
DATA_PARTITION="${DATA_PARTITION:-4096M}"
RAM="${RAM:-2G}"

SDKMANAGER="$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager"
AVDMANAGER="$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

confirm() {
    [ "${ASSUME_YES:-0}" = "1" ] && return 0
    # Asked for by hand, answered by hand. Anything driving this without a
    # terminal -- CI, an agent -- has to say --yes and mean it. Opening the
    # terminal is the test: /dev/tty passes every -r and -e there is, and then
    # fails to open when nothing is attached to it.
    if ! { exec 3</dev/tty; } 2>/dev/null; then
        echo "Not a terminal. Re-run with --yes if you really mean it." >&2
        exit 1
    fi
    printf '%s [y/N] ' "$1"
    read -r answer <&3
    exec 3<&-
    case "$answer" in [yY]*) return 0 ;; *) echo "Nothing was changed."; exit 1 ;; esac
}

size_of() { [ -e "$1" ] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"; }

status() {
    say "What is installed"
    printf '  %-28s %-8s %s\n' "cmdline-tools" "$(size_of "$ANDROID_HOME/cmdline-tools")" "$ANDROID_HOME/cmdline-tools"
    printf '  %-28s %-8s %s\n' "emulator" "$(size_of "$ANDROID_HOME/emulator")" "$ANDROID_HOME/emulator"
    printf '  %-28s %-8s %s\n' "system-images" "$(size_of "$ANDROID_HOME/system-images")" "$ANDROID_HOME/system-images"
    printf '  %-28s %-8s %s\n' "platform-tools (adb)" "$(size_of "$ANDROID_HOME/platform-tools")" "$ANDROID_HOME/platform-tools"
    printf '  %-28s %-8s %s\n' "the $AVD device" "$(size_of "$HOME/.android/avd/$AVD.avd")" "$HOME/.android/avd/$AVD.avd"
    say "Total"
    du -sch "$ANDROID_HOME" "$HOME/.android/avd" 2>/dev/null | tail -1
    if [ -x "$ANDROID_HOME/emulator/emulator" ]; then
        echo
        echo "Ready. ./dev.sh boot starts it."
    else
        echo
        echo "Not installed. ./emulator.sh install puts it here."
    fi
}

install() {
    say "Installing the simulated phone (about 5 GB, mostly the system image)"

    command -v brew >/dev/null || {
        echo "Homebrew is needed for this. https://brew.sh" >&2; exit 1
    }
    # sdkmanager and avdmanager are Java programs, and the emulator wants a JDK
    # of its own generation; 17 is what the current tools are built against.
    brew list --formula openjdk@17 >/dev/null 2>&1 || brew install openjdk@17
    brew list --cask android-commandlinetools >/dev/null 2>&1 \
        || brew install --cask android-commandlinetools

    # Licences first: sdkmanager refuses to fetch anything without them, and
    # the prompt is the whole reason an unattended install stalls.
    say "Accepting the SDK licences"
    yes | "$SDKMANAGER" --licenses >/dev/null 2>&1 || true

    say "Fetching platform-tools, the emulator, and the Android 15 image"
    "$SDKMANAGER" "platform-tools" "emulator" "$IMAGE"

    if [ -d "$HOME/.android/avd/$AVD.avd" ]; then
        echo "The $AVD device already exists; leaving it alone."
    else
        say "Creating the $AVD device"
        echo no | "$AVDMANAGER" create avd -n "$AVD" -k "$IMAGE" -d "$DEVICE"
        config="$HOME/.android/avd/$AVD.avd/config.ini"
        # Written rather than passed on the command line because avdmanager has
        # no flag for either of them.
        sed -i '' '/^disk.dataPartition.size/d;/^hw.ramSize/d' "$config"
        printf 'disk.dataPartition.size = %s\nhw.ramSize = %s\n' \
            "$DATA_PARTITION" "$RAM" >> "$config"
    fi

    status
    say "Next"
    echo "  ./dev.sh boot && ./dev.sh install && ./dev.sh pair"
    echo "  (the server must be running with --bind lan)"
}

remove() {
    local all="${1:-}"
    say "This will remove the simulated phone from this Mac"
    echo "  - the $AVD device and anything installed on it"
    echo "  - the Android 15 system image and the emulator binary"
    [ "$all" = "--all" ] && echo "  - adb, the command-line tools, and the Homebrew casks"
    echo
    echo "Nothing in this project needs them: the APK is built in CI and the"
    echo "JVM tests run without a device. ./emulator.sh install puts it back."
    confirm "Remove it?"

    # Stop it first, or the image files are removed from under a running VM.
    if [ -x "$ANDROID_HOME/platform-tools/adb" ]; then
        "$ANDROID_HOME/platform-tools/adb" emu kill >/dev/null 2>&1 || true
        "$ANDROID_HOME/platform-tools/adb" kill-server >/dev/null 2>&1 || true
    fi

    if [ -x "$AVDMANAGER" ]; then
        say "Deleting the $AVD device"
        "$AVDMANAGER" delete avd -n "$AVD" 2>/dev/null || true
    fi
    rm -rf "${HOME:?}/.android/avd/$AVD.avd" "${HOME:?}/.android/avd/$AVD.ini"

    if [ -x "$SDKMANAGER" ]; then
        say "Removing the system image and the emulator"
        "$SDKMANAGER" --uninstall "$IMAGE" "emulator" >/dev/null 2>&1 || true
    fi
    # sdkmanager leaves the directories behind often enough to check by hand.
    rm -rf "${ANDROID_HOME:?}/system-images" "${ANDROID_HOME:?}/emulator"

    if [ "$all" = "--all" ]; then
        say "Removing the command-line tools and adb"
        brew uninstall --cask android-commandlinetools 2>/dev/null || true
        brew uninstall --cask android-platform-tools 2>/dev/null || true
        echo "openjdk@17 was left alone -- other things use it."
    fi

    say "Done"
    du -sch "$ANDROID_HOME" "$HOME/.android/avd" 2>/dev/null | tail -1 || true
    echo
    echo "Put it back with: ./emulator.sh install"
}

case "${1:-status}" in
    status)  status ;;
    install) install ;;
    remove|uninstall)
        shift || true
        all=""
        for arg in "$@"; do
            case "$arg" in
                --all)     all="--all" ;;
                --yes|-y)  ASSUME_YES=1 ;;
                *) echo "unknown option: $arg" >&2; exit 2 ;;
            esac
        done
        remove "$all"
        ;;
    *)
        echo "usage: ./emulator.sh [status|install|remove [--all] [--yes]]" >&2
        exit 2
        ;;
esac
