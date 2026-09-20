# Read on Mac — the phone half

Select text anywhere on an Android phone, tap **Read on Mac**, and hear it in
your own cloned voice a second later, with ⏮ ⏯ ⏭ on the lock screen moving by
paragraph. The phone holds no model and no audio: it finds the Mac on the local
network, hands it the text, and plays the paragraphs back as the Mac makes them.

## Getting the APK onto the phone

The APK is built by GitHub Actions, not on the Mac — see
`.github/workflows/android.yml` for why. Every push to `mobile-app` publishes
one:

**On the phone**, open
`https://github.com/ThomasC998/local-TTS/releases/tag/phone-latest`
and install `read-on-mac.apk`. Android will ask once whether to allow installing
from the browser.

To build it yourself instead, open `android/` in Android Studio and
`Build → Build APK`, or run `gradle assembleDebug` in this directory.

## Pairing

1. On the Mac: `python breeze_server.py --bind lan`
2. Open `http://127.0.0.1:7860` on the Mac and go to the **Phone** panel.
3. In the app, tap **Pair**, and scan the code.

The code carries the address, the access token and the fingerprint of the Mac's
certificate. From then on the phone accepts that one certificate and no other,
so nothing else on the Wi-Fi can read or alter a read — and the token never
crossed the network, because it arrived as a photograph.

## The three ways to start a read

- **Select text → "Read on Mac"** in the selection toolbar. Works in every app,
  needs no permission. The one to use.
- **Share → Read on Mac**, for text and for screenshots.
- **The notification button**, which reads whatever is on the clipboard.

A screenshot goes to a small vision model on the Mac, which is asked for the
article and not the clock, the tab bar or the buttons. Its answer is shown for a
glance and an edit before anything is spoken; turn that off in the app if you
would rather take the chance.

## Waking the Mac

The app sends a magic packet and a connection attempt, then waits about twenty
seconds. This only works on the Mac's own network — a sleeping machine's VPN is
asleep too — and **a MacBook running on battery may not wake at all**. That is a
property of the hardware. If it matters, use *Keep on* mode before you leave:
it holds the Mac awake with the lid closed, with a battery floor, and needs
`sudo ./install_power_helper.sh` run once on the Mac.

## When something goes wrong

The app reports failures to the Mac, where they appear in the server log
prefixed `[phone]`. **Check the connection** on the app's screen says which
address answered and how long it took. For a real stack trace, plug the phone in
and run `adb logcat -s BreezeRead BreezeWake BreezeDiscovery` — that needs
`brew install --cask android-platform-tools` (about 15 MB) and nothing else.
