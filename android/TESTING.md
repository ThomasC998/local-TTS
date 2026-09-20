# Testing an Android app without an Android developer setup

Three separate problems, three different answers. Only the third one is heavy.

| I want to… | Use | Size |
|---|---|---|
| read error logs while something runs | `adb logcat` | 37 MB |
| check logic on every push | `gradle test` in CI | nothing local |
| click around a phone on the Mac | the emulator | 6.7 GB, `./emulator.sh install` |

All of it is already installed here. Nothing needs Android Studio.

---

## 1. Logs — the browser-console equivalent

`adb` is the whole of it. It works against the emulator or a real phone over
USB, with no project and no build.

```bash
adb devices                                   # what is attached
adb logcat --pid=$(adb shell pidof com.breezetts.read)   # one app's output
adb logcat -s BreezeRead:V AndroidRuntime:E   # by tag, when the app is not running
adb logcat -d -b crash                        # the last crash, with its stack
adb logcat -c                                 # clear, so the next run is clean
```

For a real phone over Wi-Fi: **Developer options → Wireless debugging**, then
`adb pair <host>:<port>` with the code shown, and `adb connect <host>:<port>`.

## 2. Driving the app without touching it

**Every trigger in an Android app is an intent, and an intent can be sent from a
shell.** That is what makes this iterable: start the thing, read the log, fix
it, repeat — no tapping.

```bash
# exactly what the selection toolbar sends
adb shell am start -a android.intent.action.PROCESS_TEXT -t text/plain \
  --es android.intent.extra.PROCESS_TEXT "'some text'" \
  -n com.breezetts.read/.ProcessTextActivity

adb shell input keyevent KEYCODE_MEDIA_NEXT   # as a headset button would
adb shell am force-stop com.breezetts.read
```

> The value is quoted **twice** — once for your shell, once for the device's,
> which re-splits it. A single set of quotes silently loses everything after the
> first space, and the intent resolves to nothing.

On a debug build, `run-as` reaches the app's own data directory, which is how
the emulator gets paired without photographing a QR code:

```bash
adb shell "run-as com.breezetts.read cat /data/data/com.breezetts.read/shared_prefs/breeze.xml"
```

**`./dev.sh` wraps all of the above** — `boot`, `install`, `pair`, `read "…"`,
`next`, `logs`, `crash`, `shutdown`. It works against a real phone too: plug it
in and skip `boot`.

## 3. Tests that need no device at all

`android/app/src/test/` runs on the JVM — a couple of seconds, no emulator, so
it runs on every push before the APK is built. Put anything there whose failure
would be *silent*: a pairing code missing its fingerprint leaves the connection
trusting any certificate, which behaves exactly like one that works.

```bash
gradle test            # in android/, needs a JDK; CI does it for you
```

The trick worth knowing: `org.json` ships with Android, and the JVM stub throws
on every call. `testImplementation("org.json:json:…")` puts the real one on the
test classpath.

Next step up, if ever needed: **Robolectric** runs the Android framework itself
on the JVM (Context, SharedPreferences, Looper) — still no device. Above that,
instrumented tests with Espresso need a device or emulator and are worth it only
for real UI flows.

## 4. The emulator, for clicking around

**Not installed by default, and removable.** `./emulator.sh install` puts it
on this Mac, `./emulator.sh remove` takes it off and gives the disk back, and
`./emulator.sh status` says which it is. `dev.sh` refuses with an explanation
rather than a stack trace when it is missing. Everything above this line — the
APK, the JVM tests — works without it.

Installed via the command-line tools only — no Android Studio:

```
openjdk@17                   305 MB   (sdkmanager is a Java program)
cmdline-tools                173 MB
emulator                     1.1 GB
system-images;android-35     3.8 GB   ← the big one
platform-tools                37 MB
                             ─────
                             5.1 GB

plus the device itself        1.6 GB   ~/.android/avd/phone.avd
                             ─────
                             6.7 GB
```

```bash
./dev.sh boot        # or: emulator -avd phone
./dev.sh shutdown
```

Two things that bit here and will bite again:

- The default AVD demands a **12 GB** userdata partition and refuses to start
  without it. `disk.dataPartition.size=4096M` in `~/.android/avd/<name>.avd/config.ini`.
- The emulator reaches your Mac's LAN address normally, so it can talk to the
  server like a real phone. (`10.0.2.2` is the host's *loopback*, if you need it.)

To make it smaller next time: an older system image is much lighter
(`system-images;android-30;google_apis;arm64-v8a` is roughly a third), and a
`google_apis` image is smaller than a `google_apis_playstore` one.

## 5. What this does not cover

Driving the UI by *looking* at it — tapping buttons, asserting on what is on
screen — is Espresso or UI Automator, or an MCP server like `mobile-mcp` that
exposes the screen to a model. None of that is installed, because for this app
the intents above reach everything the buttons do, and a screenshot is a slower
way to learn the same thing.
