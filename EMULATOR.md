# Running the phone app on the Mac

A simulated phone on this machine, with the app on it, talking to the server
here. Everything is already installed; this is how to use it.

There is no QR code in this flow — see [Pairing without a camera](#3-pairing-without-a-camera).

---

## Everything, in order

```bash
cd ~/Documents/BreezeTTS2

# 1. the server, reachable from the emulator
python breeze_server.py --bind lan

# 2. the phone (in another terminal)
cd android
./dev.sh boot        # ~1 min the first time
./dev.sh install     # downloads the latest APK from the release
./dev.sh pair        # no camera needed — see below
./dev.sh read "The harbour was quiet that morning. Nothing moved on the water."
./dev.sh logs        # ctrl-c to stop watching
```

A window opens with Android 15 in it. You can tap around in it exactly like a
phone — open the app, change settings, pull down the notification shade.

When you are done:

```bash
./dev.sh shutdown
```

---

## 1. The emulator

`./dev.sh boot` starts it and waits until Android is actually up. Or by hand:

```bash
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"
emulator -avd phone
adb devices          # should list emulator-5554
```

The device is called `phone` and its settings live in
`~/.android/avd/phone.avd/config.ini`.

## 2. Getting the app on it

```bash
./dev.sh install
```

That downloads `read-on-mac.apk` from the GitHub release and `adb install`s it.
To install a specific file instead:

```bash
adb install -r /path/to/read-on-mac.apk
```

## 3. Pairing without a camera

On your real phone you scan a QR code, because the token and the certificate
fingerprint have to reach it without crossing the network they protect. The
emulator's camera cannot photograph your Mac's screen, so `./dev.sh pair` puts
the same values in directly:

```bash
./dev.sh pair
```

It reads the pairing payload from the server here — address, token,
fingerprint, hardware address — writes it as the app's own preferences file,
and pushes it with `adb`. This works because the APK is a **debug** build, and
`run-as` lets a debug build's own data directory be written. It would not work
on a release build, which is the point.

The app then behaves exactly as a paired phone does.

## 4. Driving it without tapping

Every trigger in the app is an Android intent, so the whole read path can be
started from the terminal:

```bash
./dev.sh read "some text"   # as the selection toolbar sends it
./dev.sh next               # as a headset button sends it
./dev.sh prev
./dev.sh pause
./dev.sh stop               # force-stop the app
```

## 5. Seeing what happened

```bash
./dev.sh logs     # this app's output, live — the browser-console equivalent
./dev.sh crash    # the last crash with its stack, if there was one
```

From the Mac's side, the server's terminal shows the requests, and anything the
app failed at is posted back and appears there prefixed `[phone]`.

To see it from the server's point of view:

```bash
TOKEN=$(python3 -c "import mobile_auth; print(mobile_auth.load_token())")
curl -sk https://127.0.0.1:7860/v1/reads | python3 -m json.tool
```

## 6. Things that will bite

| Symptom | Cause |
|---|---|
| `Not enough space to create userdata partition` | The AVD wants 12 GB by default. Set `disk.dataPartition.size=4096M` in `~/.android/avd/phone.avd/config.ini`. |
| An intent loses everything after the first word | The value needs quoting **twice** — once for your shell, once for the device's. `./dev.sh read` does this for you. |
| The app says it cannot find the Mac | The server must be started with `--bind lan`, which adds the device listener on 7861. Without it there is nothing on the network to find. |
| `./dev.sh pair` fails | Either the server is not running, or the APK is a release build (`run-as` is refused). |

The emulator reaches your Mac's LAN address normally, the same as a real phone.
(`10.0.2.2` is the host's *loopback*, if you ever need that instead.)

---

More detail on testing generally — JVM tests, real phones over Wi-Fi, what the
emulator costs on disk — is in [`android/TESTING.md`](android/TESTING.md).
