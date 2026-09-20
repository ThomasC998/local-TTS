# Reading on the phone — setup and first run

Select text anywhere on an Android phone, tap **Read on Mac**, and hear it in
your own voice with ⏮ ⏯ ⏭ moving by paragraph. The phone holds no model: it
finds this Mac on your network, hands it the text, and plays back the
paragraphs as the Mac makes them.

Everything below is on your own network. No account, no cloud, nothing leaves
the house.

---

## 1. The Mac, once

```bash
cd ~/Documents/BreezeTTS2
pip install -r requirements.txt      # adds cryptography, zeroconf, qrcode
```

Then start the server so the phone can reach it:

```bash
python breeze_server.py --bind lan
```

What changes with `--bind lan`, and only then:

- it listens on the network instead of this Mac alone
- **every route requires a token**, generated on first run into
  `state/mobile_token.json` (mode `0600`, never committed)
- it serves **HTTPS** with a self-signed certificate from `state/tls/`
- it announces itself over Bonjour, so the phone still finds it after your
  router hands out a different address

Without `--bind lan` nothing changes at all: loopback only, no token, exactly as
before. macOS may ask once whether to let Python accept incoming connections —
say yes.

> The first time it starts it will also ask nothing and print the address it is
> serving on. Keep that terminal open; the server has to be running to read.

---

## 2. The power helper — optional, and what it actually does

Only needed if you want **Keep on** mode (the Mac stays awake with the lid
closed) or **Sleep when done**. Skip it and everything else still works; those
two modes just report that the helper is missing.

```bash
sudo ./install_power_helper.sh
```

It writes one file, `/etc/sudoers.d/breezetts-power`, granting your user exactly
three complete command lines without a password:

```
/usr/bin/pmset -a disablesleep 1     stop sleeping, lid closed
/usr/bin/pmset -a disablesleep 0     sleep normally again
/usr/bin/pmset sleepnow              sleep now
```

sudo matches the whole command line, so these three and nothing else. The file
is validated with `visudo -c` before it is installed, because a malformed file
in `sudoers.d` can lock you out of `sudo` entirely. No daemon, no login item,
nothing else is touched.

**The exposure:** while that file exists, anything running as you can also put
this Mac to sleep or stop it sleeping, without a password. That is the whole of
it.

**Uninstall:**

```bash
sudo ./install_power_helper.sh --remove     # or: sudo rm /etc/sudoers.d/breezetts-power
```

Also check, in **System Settings → Battery → Options**, that *Wake for network
access* is on — the phone cannot wake the Mac without it. The Phone panel tells
you whether it is.

---

## 3. The app on the phone

The APK is built by GitHub Actions, not on this Mac (Android Studio and an SDK
are ~15 GB). **On the phone**, open:

**https://github.com/ThomasC998/local-TTS/releases/tag/phone-latest**

and install `read-on-mac.apk`. Android asks once whether to allow installs from
your browser; allow it, install, then you can turn it off again.

*(To rebuild it: push to the `mobile-app` branch, or run the "Android app"
workflow by hand. To build locally instead, open `android/` in Android Studio.)*

---

## 4. Pairing, once

1. On the Mac, open <https://127.0.0.1:7860> and go to the **Phone** tab.
   Your browser will warn about the self-signed certificate — continue; it is
   the certificate this Mac just made for itself.
2. In the app, tap **Pair by scanning the Mac's code**, allow the camera, and
   scan.

The code carries the address, the access token, and the fingerprint of the
Mac's certificate. From then on the phone accepts **that one certificate and no
other**, so nothing else on the Wi-Fi can read or alter a read — and the token
never crossed the network, because it arrived as a photograph.

**To un-pair everything:** Phone tab → *Forget paired devices*, then restart the
server. Every paired phone stops working until it is paired again.

---

## 5. Test it, in order

Phone and Mac on the same Wi-Fi. Server running.

1. **Connection.** Open the app → **Check the connection**. Expect
   *"Thomass-MacBook-Pro answered at 192.168.x.x in 40 ms"*. If it does not,
   see the troubleshooting table below.

2. **Selected text.** Open any web page on the phone, long-press a paragraph to
   select it, and choose **Read on Mac** from the selection toolbar (it may be
   behind the ⋮ at the end). Audio should start within a few seconds.

3. **The transport.** While it plays, pull down the notification shade. Tap
   ⏭ — it waits two seconds (so holding it scrolls instead of synthesizing
   everything on the way) and then starts the next paragraph. Tap ⏮ — the
   previous paragraph replays **instantly**, because it is already made. Lock
   the phone and try the same buttons from the lock screen.

4. **The share sheet.** From any app: **Share → Read on Mac**.

5. **The clipboard button.** Copy some text, pull down the shade, tap **Read
   the clipboard** on the ongoing *Read on Mac* notification.

6. **A screenshot.** See the next section first, then: take a screenshot, crop
   it in Android's own screenshot editor if you like, **Share → Read on Mac**.
   The text appears for a look and an edit before anything is spoken.

7. **Sleep.** In the app, set **Sleep when done**, finish a read, and the Mac
   sleeps two minutes later. Then wake it from the phone by starting another
   read — see the caveat below.

---

## 6. Screenshots

A screenshot is mostly not prose: a clock, a battery icon, a nav bar, buttons.
Read it in order and the first thing you hear is "9:41". So a small multimodal
model is asked for the article and told to leave the interface out.

**Set it up:**

1. Open **LM Studio** → **Developer** tab → start the local server (port 1234).
2. Load a model with the **vision** badge. A **Qwen-VL** (e.g.
   `qwen2.5-vl-7b-instruct`) is the best of the local ones at reading screens.
3. Check the **Phone** tab in the web UI: *Screenshots* should say
   `Ready — lmstudio · <model>`.

Optional, in `.env` — only if LM Studio is somewhere unusual or you want to pin
a specific model:

```
BREEZE_VISION_BASE_URL=http://127.0.0.1:1234/v1
BREEZE_VISION_MODEL=qwen2.5-vl-7b-instruct
BREEZE_VISION_ENABLED=1
```

**Without LM Studio running** it still works, using macOS's own text
recognition — accurate, but with no opinion about what matters, so the clock
and the nav bar come through too. The app says so and shows you the text to
edit first. Two ways to narrow it down:

- crop the screenshot in Android's screenshot editor *before* sharing
- delete the stray lines on the confirm screen *after*

---

## 7. Waking the Mac — read this before relying on it

The app sends a wake-on-LAN magic packet and a connection attempt, then waits
about twenty seconds.

- It only works **on the Mac's own network**. A sleeping Mac's VPN is asleep
  too, so this can never work from outside the house.
- **A MacBook on battery may not wake at all.** That is the hardware, not the
  app. Measure it on yours: close the lid, wait for it to sleep, and try a read
  — once on the charger, once on battery.

If it does not wake on battery, use **Keep on** mode before you leave: it holds
the Mac awake with the lid closed, gives up below 25% battery, and releases
after 8 hours regardless.

---

## 8. When something goes wrong

| What you see | What it means |
|---|---|
| *"…did not answer"* | Different Wi-Fi, server not running, or asleep and not woken. Check the terminal. |
| *"The Mac did not recognise this phone"* | The token was rotated. Pair again. |
| *"This is not the Mac this phone was paired with"* | The certificate changed (the Mac's address changed). Pair again. |
| *"needs the power helper"* | Section 2, or use a different sleep mode. |
| *"No local vision model"* | Section 6 — LM Studio is not running and pyobjc is not installed. |
| Nothing at all | The app reports failures to the Mac: look for `[phone]` in the server terminal. |

The app's **Diagnostics** block shows the fingerprint, the address that last
answered, and where wake packets go. For a real stack trace, plug the phone in
and run `adb logcat -s BreezeRead BreezeWake BreezeDiscovery` — that needs only
`brew install --cask android-platform-tools` (~15 MB), no SDK and no emulator.

---

## 9. Removing it

- **The app:** uninstall it on the phone as usual.
- **The pairing:** Phone tab → *Forget paired devices*. Or delete
  `state/mobile_token.json` and `state/tls/`.
- **The power helper:** `sudo ./install_power_helper.sh --remove`.
- **Network exposure:** start the server without `--bind lan`. It goes back to
  this Mac only, with no token and no certificate, exactly as it was.
