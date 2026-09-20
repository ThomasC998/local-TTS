#!/bin/bash
#
# Driving the phone app from the terminal: boot a simulated phone, install the
# APK, pair it, start a read, and read the logs -- with no tapping at all.
#
#     ./dev.sh boot                 start the emulator and wait for it
#     ./dev.sh install              fetch the latest APK and install it
#     ./dev.sh pair                 pair it with the server running here
#     ./dev.sh read "some text"     start a read, as the selection toolbar would
#     ./dev.sh next | prev | stop   the transport, as a headset would
#     ./dev.sh logs                 follow this app's log (ctrl-c to stop)
#     ./dev.sh crash                the last crash, if there was one
#     ./dev.sh shutdown             stop the emulator
#
# Why this exists: every trigger in this app is an Android intent, and an
# intent can be sent from a shell. So the whole read path -- find the Mac,
# start the read, play the paragraphs -- can be exercised and watched without
# touching the screen, which is what makes it something you can iterate on.
#
# It works against a real phone too: plug it in and skip `boot`.

set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}"
export ANDROID_HOME="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}"
export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"

APP="com.breezetts.read"
AVD="${AVD:-phone}"
REPO="${REPO:-ThomasC998/local-TTS}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"

case "${1:-}" in

boot)
    if adb shell true 2>/dev/null; then
        echo "A device is already attached."; exit 0
    fi
    echo "Starting $AVD…"
    nohup emulator -avd "$AVD" -no-snapshot-save -no-boot-anim \
        >"${TMPDIR:-/tmp}/emulator.log" 2>&1 &
    until [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do
        sleep 5
    done
    echo "Booted: Android $(adb shell getprop ro.build.version.release | tr -d '\r')"
    ;;

install)
    cd "${TMPDIR:-/tmp}"
    gh release download phone-latest --repo "$REPO" --pattern 'read-on-mac.apk' --clobber
    adb install -r read-on-mac.apk
    adb shell pm grant "$APP" android.permission.POST_NOTIFICATIONS 2>/dev/null || true
    echo "Installed."
    ;;

pair)
    # The pairing code is meant to be photographed, and an emulator's camera
    # cannot photograph the Mac's screen. A debug build lets its own data
    # directory be written through run-as, so the same values go in directly.
    python3 - "$PROJECT" <<'PY' >"${TMPDIR:-/tmp}/breeze.xml"
import html, json, sys, os
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])
import mac_power, mobile_auth
payload = mobile_auth.pairing_payload(7860, {"mac_addresses": mac_power.mac_addresses()})
if not payload.get("host"):
    raise SystemExit("This Mac has no network address; start the server with --bind lan")
print('<?xml version="1.0" encoding="utf-8" standalone="yes" ?>')
print("<map>")
for key, value in (
    ("host", payload["host"]),
    ("token", payload["token"]),
    ("fingerprint", payload["fingerprint"]),
    ("name", payload["name"]),
    ("macs", ",".join(payload["mac_addresses"][:1])),
):
    print(f'    <string name="{key}">{html.escape(str(value))}</string>')
print(f'    <int name="port" value="{payload["port"]}" />')
print('    <boolean name="confirmScreenshots" value="false" />')
print("</map>")
PY
    adb shell "run-as $APP mkdir -p /data/data/$APP/shared_prefs"
    adb push "${TMPDIR:-/tmp}/breeze.xml" /data/local/tmp/breeze.xml >/dev/null
    adb shell "run-as $APP cp /data/local/tmp/breeze.xml /data/data/$APP/shared_prefs/breeze.xml"
    adb shell am force-stop "$APP"
    echo "Paired with $(adb shell "run-as $APP cat /data/data/$APP/shared_prefs/breeze.xml" | sed -n 's/.*name="host">\([^<]*\).*/\1/p')"
    ;;

read)
    text="${2:?usage: ./dev.sh read \"some text\"}"
    adb logcat -c
    # Quoted twice on purpose: once for this shell, once for the device's.
    adb shell am start -a android.intent.action.PROCESS_TEXT -t text/plain \
        --es android.intent.extra.PROCESS_TEXT "'$text'" \
        -n "$APP/.ProcessTextActivity" >/dev/null
    echo "Started. ./dev.sh logs to watch it."
    ;;

next)    adb shell input keyevent KEYCODE_MEDIA_NEXT ;;
prev)    adb shell input keyevent KEYCODE_MEDIA_PREVIOUS ;;
pause)   adb shell input keyevent KEYCODE_MEDIA_PLAY_PAUSE ;;
stop)    adb shell am force-stop "$APP" ;;

logs)
    # Everything the app's own process says, which includes the exceptions it
    # swallows into toasts -- the equivalent of a browser console for it.
    pid="$(adb shell pidof "$APP" 2>/dev/null | tr -d '\r')"
    if [ -n "$pid" ]; then adb logcat --pid="$pid"; else adb logcat -s BreezeRead:V BreezeWake:V BreezeDiscovery:V AndroidRuntime:E; fi
    ;;

crash)   adb logcat -d -b crash | tail -40 ;;

shutdown)
    adb emu kill 2>/dev/null || pkill -f qemu-system || true
    echo "Emulator stopped."
    ;;

*)
    sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
