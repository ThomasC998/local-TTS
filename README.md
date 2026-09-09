# Breeze TTS 2, on a Mac or a Windows PC

A local text-to-speech server: copy text anywhere, press a key, hear it read in
a voice you designed or cloned. It runs entirely on your own machine — the model,
the audio, the voice library, the history.

The same web interface, the same shortcuts and the same features on both
platforms. What differs is only what is underneath: MLX on Apple Silicon, CUDA
on Windows, Hammerspoon or a small Python daemon for the keyboard hook. None of
that is visible while using it.

**Requirements**

| | macOS | Windows 11 |
| :--- | :--- | :--- |
| Hardware | Apple Silicon (M1 or newer) | An NVIDIA GPU, 8 GB VRAM or more |
| Model on disk | 3.5 GB | 5.9 GB (INT8) or 7 GB (BF16) |
| Python | 3.10 – 3.12 | 3.10 – 3.12 |
| Also needed | [SoX](http://sox.sourceforge.net) — the installer offers it | SoX, likewise |

There is no CPU mode. This model generates several times slower than the speech
it produces on a CPU, so the audio would stutter continuously; the server says
so and refuses to start rather than appearing to work.

## Install

```bash
git clone <your-repository-url> BreezeTTS2
cd BreezeTTS2
```

**macOS**

```bash
./install.sh
```

**Windows** (PowerShell, in the project folder)

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

Either one creates a virtual environment, installs the right build of PyTorch
for your machine, offers to install SoX, copies `.env.example` to `.env`, and
downloads the speech model (3.5 GB on a Mac, 5.9 GB on Windows). It is safe to
run again; the model download resumes where it stopped.

Then:

```bash
python breeze_server.py
```

and open <http://127.0.0.1:7860>.

If you would rather not use a virtual environment — inside an existing conda
environment, say — pass `--no-venv` or `-NoVenv`. See [MODELS.md](MODELS.md) if
you want a different checkpoint, and [Configuration](#configuration) for the
settings.

## Quick start

Once a voice exists (see [the walkthrough](#from-nothing-to-a-hotkey-that-reads-your-clipboard)),
any code on the machine can call the server:

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from my laptop.","instruction":"A warm narrator.","cfg_scale":4}' \
  -o hello.wav
```

See `client_example.py` for Python clients covering WAV, streaming PCM, voice
cloning, and the LLM-to-TTS loop.

## What this can do

| Feature | Where you use it | Section |
| :--- | :--- | :--- |
| Synthesize text to a WAV | `POST /v1/audio/speech`, or the **Synthesize** tab | [Quick start](#quick-start) |
| Stream audio with live timing | `format: "sse"`, or any tab with playback | [Streaming](#streaming-and-playback) |
| Invent a voice from a description | **Design a voice** tab | [Voices](#voices-and-why-a-seed-is-not-one) |
| Clone a voice from a recording | **Design a voice** tab → From a recording | [Voices](#voices-and-why-a-seed-is-not-one) |
| Keep and manage saved voices | **Voice library** tab | [Voices](#voices-and-why-a-seed-is-not-one) |
| Check a voice holds on new text | **Test a voice** tab | [Voices](#voices-and-why-a-seed-is-not-one) |
| Give one voice several tones | **Voice library** → a voice → Add a variation | [Several tones](#several-tones-for-one-voice) |
| Rotate tone through a long read | Automatic; tuned per voice or globally | [Several tones](#several-tones-for-one-voice) |
| Speak the clipboard from any app | `ctrl+alt+S` | [Speak anything you copy](#speak-anything-you-copy) |
| Rewrite text with an LLM, then speak | `ctrl+alt+A` | [Streaming the model](#streaming-the-language-model-into-the-engine) |
| Change the shortcuts | **System speech** tab → Shortcuts | [Speak anything you copy](#speak-anything-you-copy) |
| Pick the hotkey voice and settings | **System speech** tab | [Speak anything you copy](#speak-anything-you-copy) |
| Edit the LLM prompt | **System speech** tab → Prompt | [Streaming the model](#streaming-the-language-model-into-the-engine) |
| Review everything ever spoken | **System speech** tab → Recent | [Everything spoken is kept](#everything-spoken-is-kept) |
| Clean up text without audio | `POST /v1/text/prepare` | [Text preparation](#text-preparation) |

The web UI at <http://127.0.0.1:7860> has five tabs, in the order you would use
them: **Synthesize** (one-off speech), **Design a voice** (invent or clone),
**Voice library** (keep, edit, add tones), **Test a voice** (does it hold?), and
**System speech** (the hotkey path).

## From nothing to a hotkey that reads your clipboard

The whole path, in order. It takes one pass through the web UI and one command.

**1. Start the server and open the UI.**

```bash
python breeze_server.py
```

Then open <http://127.0.0.1:7860>.

**2. Make a voice.** On **Design a voice**, describe the speaker you want
("a calm, unhurried woman in her thirties, warm and a little dry") and generate
five takes. Each take is a *different* voice on its own seed. Play them, pick
one, name it, save it. Or switch to *Clone from a recording* to clone a real
voice — that needs the recording plus the exact words spoken in it, because this
runtime has no speech recognition and a wrong transcript clones badly.

Cloning has a shortcut worth knowing: **Save this recording as a voice**, in the
clone panel. A cloned voice is conditioned on the recording every time it
speaks, never on the take, so the takes are an audition rather than the voice
itself. If you already trust the recording, that button saves it directly — file,
transcript and voice direction — with nothing generated at all. Takes can still
be generated afterwards and added as tone variations.

Aim for a take of **15 seconds or more**. That clip becomes the reference every
later request is anchored to, and short references clone weakly.

**3. Check it holds.** On **Test a voice**, run three or four sentences it has
never spoken. With *Anchor to saved sample* the speaker should stay put. This
tab exists because the intuitive alternative — save the seed, replay it — does
not work, and it is worth hearing why once.

**4. Give it more than one tone.** On **Voice library**, open the voice and
expand *References and tone rotation* → **Add a variation**. Describe the
difference ("warmer and slower", "brighter, smiling, a little quicker"), generate
two takes, keep the one that fits with a label and tags. Now a long read has
somewhere to move to. Repeat for as many tones as you want.

**5. Point the hotkey at it.** On **System speech**, choose the voice, set the
rotation you want, and press *Speak the clipboard* to hear it work.

**6. Install the hotkeys.**

```bash
python install_hotkeys.py
```

On macOS this sets up Hammerspoon; on Windows it sets up the Python hotkey
daemon. Either way, copy a paragraph anywhere and press `ctrl+alt+S`. Press
`ctrl+alt+A` instead to run it through the language model first, which lays the
text out for the ear before it is spoken. `ctrl+alt+S` again stops it;
`ctrl+alt+→` and `ctrl+alt+←` move a paragraph at a time.

**7. Make it survive a reboot** (optional).

```bash
python install_hotkeys.py --startup
```

## What is installed

Everything the two platforms share is in one file each. Where they differ, the
difference is in a directory named for it.

| Piece | Location |
| :--- | :--- |
| HTTP server and web UI | `breeze_server.py`, `web/index.html` |
| Persistent engine, chunking, voice lock | `breeze_pipeline.py` |
| Seekable read session | `speech_session.py` |
| LLM-to-speech pipeline | `speech_pipeline.py` |
| Local playback | `audio_out.py` |
| Voice library | `voice_store.py` |
| Utterance archive | `archive.py` |
| System-speech settings | `speech_config.py` |
| **Which speech runtime, per machine** | `tts_backends/` |
| MLX runtime, Apple Silicon (Apache-2.0) | `breeze-tts-mlx/` |
| PyTorch runtime, CUDA | `breeze-tts-torch/` |
| **Which language model, per configuration** | `llm_providers/` |
| **The clipboard, device watching, autostart** | `platform_support.py`, `audio_out.py` |
| **The shortcuts, and both hotkey hosts** | `hotkeys/` |
| Installers | `install.sh`, `install.ps1`, `download_model.py`, `install_hotkeys.py` |
| Configuration | `.env`, `state/system_speech.json` |
| Checkpoint | `chkpt-mlx-int8/` or `chkpt-breeze-tts-2-int8/` — not in the repository |

The four bold rows are the whole of the platform difference. Everything else is
the same code running on both machines.

## Configuration

Two files, and they hold different kinds of thing.

**`.env`** — what is true about *this machine*: which language-model provider,
which API key, where the checkpoint is. It is not committed, so every machine
keeps its own. Copy `.env.example`, which documents every setting, and edit it.
Missing settings fall back to defaults, so a machine happy with the defaults
does not need the file.

**The System speech tab** — what is true about *how you want it to read*: the
voice, the shortcuts, tone rotation, the model prompt, the archive, the output
device. Stored in `state/system_speech.json`, edited in the browser, never by
hand.

### Choosing a language model

The language-model pass is optional — speech works without it — but it is what
makes a copied web page read like prose instead of like a page. Four providers
are built in:

| `BREEZE_LLM_PROVIDER` | Needs | Good for |
| :--- | :--- | :--- |
| `gemini` | `GEMINI_API_KEY` from [AI Studio](https://aistudio.google.com/apikey) | The default. One key, nothing to install. |
| `openrouter_free` | `OPENROUTER_API_KEY` from [OpenRouter](https://openrouter.ai/keys) | No cost. The model is picked from whatever is free at start-up. |
| `openrouter_paid` | The same key, plus credit | A model you choose and can rely on. Set `BREEZE_OPENROUTER_MODEL`. |
| `vertex` | The gcloud SDK and `gcloud auth application-default login` | A machine already set up for Google Cloud. No API key at all. |

Left unset, the first provider with a credential present is used, API keys
before Vertex.

The free OpenRouter models change from week to week, so nothing is hard-coded:
the catalogue is fetched at start-up and the free model with the longest context
wins. To see what is free right now, and to prefer particular ones:

```bash
python -m llm_providers.openrouter --list-free
# then, in .env:
BREEZE_OPENROUTER_PREFER=llama,qwen
```

`llm_providers/openrouter.py` imports nothing from this project and can be
lifted into another one as it stands.

**Adding a provider** takes one file. Subclass `Provider` in
`llm_providers/base.py` — four methods — and add a line to `REGISTRY`. The
settings page, the status endpoint and the `.env` documentation are all
generated from the registry, so there is nothing else to change.

### Changing the shortcuts

On the **System speech** tab, under *Shortcuts*: click a box and press the keys.
Both hotkey hosts read the result from the server, so a change applies on
whichever machine is asking. Saving does not rebind a hook that is already
running — reload Hammerspoon (`open -g "hammerspoon://breeze-reload"`) or restart
the daemon.

Every shortcut needs at least one modifier; a bare letter would fire while you
were typing it. Two actions cannot share a combination, and that is rejected
rather than resolved — whichever one won would depend on registration order.

## API

The server documents itself. `GET /v1/capabilities` returns the full parameter
table — every field, its type, its default and an example — along with which
speech backend and language-model provider are live on this machine. That is
generated from the same table the code validates against, so it cannot drift out
of date the way a written reference does.

```bash
curl -s http://127.0.0.1:7860/v1/capabilities | python -m json.tool
```

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Readiness, model path, sample rate |
| `GET` | `/v1/capabilities` | Parameter reference as JSON |
| `POST` | `/v1/text/prepare` | Clean up text with the language model, no audio |
| `POST` | `/v1/audio/speech` | Synthesize speech |
| `POST` | `/v1/voice-previews/design` | Invent voices from a description |
| `POST` | `/v1/voice-previews/clone` | Clone a voice from a recording |
| `POST` | `/v1/voice-previews/{id}/save` | Keep a candidate as a voice |
| `GET` | `/v1/voices` | The saved-voice library (plus CRUD) |
| `POST` | `/v1/text-to-speech/{voice_id}` | Synthesize with a saved voice |
| `POST` | `/v1/speak` | Speak text (or the clipboard) on this machine |
| `POST` | `/v1/speak/stop` | Silence the current utterance |
| `GET` | `/v1/speak/status` | What is speaking, and how well |
| `GET` | `/v1/voices/{id}/references` | A voice's reference recordings (plus CRUD) |
| `POST` | `/v1/voices/{id}/variations` | Audition the same voice under a new direction |
| `GET` | `/v1/system-speech/config` | Hotkey settings, incl. the LLM prompt |
| `GET` | `/v1/system-speech/history` | Everything spoken, with audio |

The parameter table also lives in `breeze_server.py` as `PARAMETERS`, with an
explanation and an example value for every field, and is served verbatim from
`/v1/capabilities`.

## Streaming and playback

Audio is available three ways, and every one of them is the same samples:

| Where | How | Use it for |
| :--- | :--- | :--- |
| A file | default, or `format: "wav"` | scripting, archiving |
| Raw PCM | `format: "pcm"`, or `/stream` | lowest latency into your own player |
| Events | `format: "sse"` | playback with live timing and boundaries |

The SSE stream carries base64 PCM plus a realtime factor measured so far and a
`boundary` event at the end of each sentence — the only place a player can pause
to rebuild its buffer without the listener hearing a glitch. The web UI's player
uses all of it, and `client_example.py` is a working client for each of the
four ways to get audio out.

`POST /v1/speak` is the fourth way: the server plays the audio itself, on this
machine's speakers. That is what the hotkey uses.

### Sample rates, and the crackle that comes from getting them wrong

The engine produces **24 kHz**. Almost no output device runs at 24 kHz — the
headphones on this machine run at 44.1 kHz — so something has to resample. Doing
that *per block* is the one thing that must not happen: a resampler that cannot
see across a block edge leaves a discontinuity at every one of them, which is
heard as continuous crackling. Measured here, block-wise conversion of a signal
peaking at 0.4 differs from a continuous conversion by up to **0.28**.

The tell is unmistakable: the streamed audio crackles while the WAV written from
the very same samples is clean, because the file is converted in one pass.

This has now been fixed twice, in both playback paths, the same way:

- **In the browser**, `AudioContext` is pinned to the stream's own rate and a
  single `AudioWorkletNode` pulls from a ring buffer, so no per-block scheduling
  or resampling remains.
- **On the server**, the output stream is opened at the *device's own* rate —
  leaving PortAudio nothing to convert — and the audio goes through one `soxr`
  resampler that lives for the whole utterance and carries its state across
  blocks. `test_system_speech.py` asserts the result is bit-identical to a
  one-shot conversion of the same signal.

Every other discontinuity the player can create is ramped rather than cut: the
start of an utterance, the end, a stop mid-word, and the moment the engine falls
behind. A jump from a mid-waveform sample to silence is a click in its own right.

If you ever hear it again, the **System speech** tab shows the rate pair and the
underrun count live (`24 → 44.1 kHz`). Periodic crackle points at a rate problem;
random dropouts with a rising underrun count mean the engine is genuinely behind,
and a larger safety buffer is the fix.

One unrelated cause worth knowing: if a Bluetooth headset is also the *input*
device and anything opens the microphone, macOS drops it from A2DP to hands-free
mode, and everything becomes narrowband and noisy. That affects all audio on the
machine, including the saved file, which is how you tell the two apart.

## Speak anything you copy

Copy text in any application, press a hotkey, and it is read aloud.

| Keys | What happens |
| :--- | :--- |
| `ctrl+alt+S` | Speak the clipboard — **press again to stop** |
| `ctrl+alt+A` | The same, through the language model first |
| `ctrl+alt+→` | Skip to the next paragraph (hold to scroll) |
| `ctrl+alt+←` | Go back a paragraph |
| `ctrl+alt+X` | Stop |

Those are the defaults, and they are the same on both platforms. Change them on
the **System speech** tab; see [Changing the shortcuts](#changing-the-shortcuts).

`ctrl+alt+S` and `ctrl+alt+A` are toggles, and that is the whole gesture for
reading something new: press once to silence what is playing, copy the next
thing, press again. Nothing has to be timed and there is no second key to
remember.

The skip keys are arrows because arrows sit in the same place on every layout --
brackets do not, and on AZERTY they need a shift combination. `ctrl+alt` is free
on both systems: macOS reserves Command combinations, Windows reserves Windows-key
ones, and neither reserves this pair broadly. The one known collision is
Rectangle on macOS, which claims `⌃⌥` with the arrows in its *default* shortcut
set — turn on its alternate shortcuts, or pick different seek keys.

Install it once:

```bash
python install_hotkeys.py             # set up whichever host this platform uses
python install_hotkeys.py --check     # report only: is anything missing?
python install_hotkeys.py --startup   # also start at login
```

Everything else is configured on the **System speech** tab of the web UI: which
voice, tone rotation, the model prompt, archiving, output device.

### Moving through a read, a paragraph at a time

A paragraph -- a block of text with a blank line before and after it -- is the
unit the skip keys move through. It is also what the language model is asked to
produce: one main idea per paragraph, usually five or six sentences, longer when
one idea genuinely needs it.

Each press cancels the generation in flight and moves a target; the engine only
starts on where you landed once the presses stop for **two seconds**. So holding
the next-paragraph key scrolls the document rather than synthesizing every
paragraph on the way past, and nothing is generated that will not be heard. Everything already
produced up to the moment of the press is kept in the archive.

Backwards works the same way, and works on paragraphs already spoken: the text
is kept, so an earlier paragraph is simply synthesized again. On the model path
the rewrite runs ahead of playback to the end of the document, which is what
makes going back possible at all.

The memory cost of all this is nil. Each paragraph is its own generation, so
abandoning one closes its codec request, releases the engine gate and trims the
allocator pools on the way out -- the same lifecycle a web-UI request gets.
Skipping through twenty paragraphs costs no more than speaking one.

### Losing the output device mid-sentence

Bluetooth headphones run out of battery, macOS moves the system output to the
laptop speakers, and a private document is suddenly being read to the room. So
the server watches for it: when the default output device changes, or the device
a stream was opened on goes away, playback **stops where it is** and the read is
left paused -- the same thing a video player does.

A paragraph press picks it up again. The first press after a pause resumes at
the paragraph it stopped in rather than moving, because that paragraph was cut
off part way through and has to be spoken from its start regardless. Press twice
and the second press moves on.

The device is asked of CoreAudio directly rather than of PortAudio, whose device
list is a snapshot taken when it was initialised -- a device that disconnected
two seconds ago is still in it. Turn the behaviour off with *Pause when the
output device changes* on the System speech tab if you would rather the audio
follow the system.

## Surviving a reboot

Two separate things have to be running, and neither starts on its own:

| Piece | Why |
| :--- | :--- |
| The Breeze server | Holds the model and does the speaking |
| The hotkey host | Owns the keyboard hook and the clipboard read |

One flag sets up both, on either platform:

```bash
python install_hotkeys.py --startup            # install both, then report
python install_hotkeys.py --check              # what would survive a reboot now
python install_hotkeys.py --uninstall-startup
```

Without `--startup` nothing is added to your login items. Installing one because
a script happened to be run is exactly what an installer should ask about first.

The server holds the checkpoint resident for as long as it runs, so starting it
at login is a standing memory cost — 3 GB of unified memory on a Mac, about
5.2 GB of VRAM on Windows. If that is not wanted, leave it out and start the server by
hand; the hotkeys report "Breeze server is not running" rather than failing
silently.

### On macOS

A LaunchAgent at `~/Library/LaunchAgents/com.breeze.tts.server.plist` runs the
server in the login session, which is what lets it read the clipboard and open
an output device. Hammerspoon starts itself, via `hs.autoLaunch(true)` in the
installed `breeze.lua`. Three details in the plist are load-bearing:

* **`EnvironmentVariables`.** launchd starts a process with almost no `PATH`, so
  `gcloud` would not be found and a Vertex project could not be resolved. The
  agent gets Homebrew on its path, the project's `.env`, and any `GOOGLE_*`
  variables that were set when it was installed. The credentials themselves need
  nothing: Application Default Credentials are a file under `~/.config/gcloud`
  that the agent, being the same user, already reads.
* **`KeepAlive: {SuccessfulExit: false}`.** Restarts after a crash, but not after
  a clean exit -- so stopping the server by hand keeps it stopped.
* **`ThrottleInterval: 30`.** The checkpoint takes the better part of a minute to
  load. Without this, launchd's restart throttle fights the model load.

Accessibility permission for Hammerspoon persists across reboots and does not
need re-granting.

### On Windows

Two shortcuts in the user's Startup folder — one for the server, one for the
hotkey daemon. Both run under `pythonw.exe` rather than `python.exe`, so no
console window stays on screen for the session.

The daemon starts with `--wait-for-server 120`, because at login the two start
at the same moment and the daemon wants the configured shortcuts, which only the
server can supply. While it waits it binds the defaults, so the keys work even
if the server never comes up.

Nothing needs administrator rights, and nothing is written outside your own user
profile.

### Why a hotkey host at all, and not a browser tab

The trigger has to do two things a web page cannot. It has to register a
system-wide hotkey, and it has to read the clipboard while some *other*
application is frontmost -- `navigator.clipboard.readText()` requires the
document to have focus, so it throws in exactly the situation this feature
exists for.

Everything past that point runs in the server: it resolves the voice, calls the
model, synthesizes, and plays the audio on its own output device. So the trigger
stays a single HTTP POST with no audio code in it, and nothing has to be
configured in two places -- which is also why the two platforms can use
completely different hotkey hosts without any other difference between them.

macOS uses Hammerspoon because it is already there for most people who want this
feature, and it solves starting at login and surviving display sleep. Windows
has no equivalent, so `hotkeys/daemon.py` does the same job in about the same
amount of code, using `pynput` for the hook. It works on macOS too, for anyone
who would rather not install Hammerspoon.

If the hotkey does nothing, `install_hotkeys.py --check` names which of the usual
causes it is.

## Several tones for one voice

A two-page article read in one unvarying tone gets tiring. So a voice holds a
*list* of reference recordings rather than a single one -- the same speaker
under different directions -- and a long read rotates between them.

Build a variation from the **Voice library** tab: open a voice, describe how the
take should differ ("warmer and slower", "brighter, smiling"), generate, and keep
the one that fits with a label and tags. The voice's own recording holds the
identity while the description moves only the delivery, which is `edit` mode --
so a variation is still recognisably the same person.

Rotation switches every 1000 words by default, in random order, and **only at a
paragraph start**, so the change is never audible inside a sentence. Text with no
blank lines in it has no paragraph starts to use, so sentence starts stand in
automatically -- otherwise the most common input of all, a passage copied out of
an article, would silently never rotate.

Every reference carries the exact transcript of its own audio. That is not
bookkeeping: the model conditions on the recording and its words together, so a
reference without a transcript is rejected at write time rather than discovered
as a bad clone later.

## Streaming the language model into the engine

`ctrl+alt+A` sends the text to the configured language model and speaks the
reply *as it is written* -- the first sentence is already playing while the rest
is still being generated. Which model that is does not matter here; every
provider in `llm_providers` streams, and this pipeline consumes deltas.

The prompt's job is pacing: blank lines where the text moves to a new point,
which become an audible pause so the listener has a moment to absorb what was
just said. It is editable on the System speech tab and stored in
`state/system_speech.json`, so iterating on it never means editing Python.

Three things make the hand-off robust:

- **No backpressure on the model.** Text is small, so the queue between the two
  is unbounded and the producer never blocks. That removes the failure this
  pipeline would otherwise be prone to: stalling the HTTPS connection to the
  provider because synthesis fell behind.
- **The consumer waits rather than dying.** The engine holds its gate across a
  gap so the utterance stays contiguous, and gives up only after 45 seconds of
  silence -- which ends the utterance cleanly instead of hanging.
- **A stall watchdog.** If the model goes quiet partway through a sentence, that
  fragment is spoken after 3 seconds rather than sitting unsaid.

Output is plain text, not a structured response: JSON escaping and partial-JSON
parsing would add latency and failure modes at precisely the wrong point. Instead
the reply must open with a `<<<SPEAK>>>` marker and everything before it is
discarded, which is what stops a stray "Sure, here you go:" from being read
aloud. A sanitizer then strips markdown and any invented bracket tag from each
chunk on its way into the engine.

Authentication is Application Default Credentials only. The project comes from
the environment, then the gcloud CLI's active config, then the ADC file itself,
so a machine that can run `gcloud auth application-default login` needs no
configuration here. The model is the first of `gemini-3.5-flash-lite`,
`gemini-3.1-flash-lite`, `gemini-2.5-flash-lite` that the project can call.

## Everything spoken is kept

With archiving on (the default), every utterance leaves a directory under
`state/archive/<date>/<id>/`: the input text, the model's raw reply, what the
engine was actually given, the audio, and a `meta.json` recording the voice, the
model, the seed, and which reference spoke which stretch. An append-only
`index.jsonl` indexes them, and the newest 500 are kept.

Per-utterance files plus an append-only index, rather than one large JSON that
every generation rewrites: appending cannot corrupt what is already there, and a
half-written entry costs one utterance instead of the whole history.

### When the recordings pile up

Every utterance is archived: the text, and the audio. They age differently --
a thousand utterances of text is a few megabytes, a hundred of audio is a
gigabyte -- so past **500 MB** of audio a banner appears above the tabs, on
every tab, with a button that zips the recordings into `state/exports/` and
clears them from the archive.

The written history stays. Only `audio.wav` is removed; the input text, the
model's output, what was actually spoken and the metadata are untouched, and
each entry records which zip its recording went into. The zip keeps each
recording's archive path and carries a `manifest.json`, so it is still readable
months later, and nothing is deleted until the zip has been reopened and
verified to contain every file.

The size is measured when the server starts and once an hour while it runs, so
the warning finds you rather than waiting to be looked for.
`BREEZE_ARCHIVE_AUDIO_WARN_MB` moves the threshold.

## Voices, and why a seed is not one

The browser UI has four tabs: **Synthesize** (single takes, as before),
**Design a voice** (one script, up to 20 takes, keep the one you like),
**Voice library** (search, edit, favourite, delete), and **Test a voice**
(speak several new lines and check they still sound like the saved sample).

A saved voice stores a reference recording, its exact transcript, and every
generation setting. That recording is the part that matters:

- **A seed does not carry a voice onto different words.** It fixes the sampling,
  not the speaker — with no reference, the model invents a voice from the
  instruction *and the text in front of it*.
- Measured here, in an MFCC speaker distance where an unrelated voice sits at
  `0.042`: replaying a saved voice's seed and settings on new text landed at
  `0.010`–`0.024`, while anchoring to its recording landed at `0.004`.

Two consequences, both on by default:

- `voice_mode=anchor` conditions every request on the voice's stored recording.
- `voice_lock=true` fixes the "every sentence sounds different" problem in long
  text: the first chunk is generated normally, then becomes the reference for
  the rest. Drift between the halves of a four-sentence paragraph fell from
  `0.019` to `0.006`. `python test_voice_lock.py` pins that logic.

Every tab plays audio as it is generated, streaming from the first chunk, and
keeps the finished file for download. In the Design and Test tabs the next take
does not start until the previous one has finished playing, so takes never talk
over each other.

## Memory

`GET /health` reports what the process holds, in whichever allocator's terms
this machine uses — MLX's pools and the MPS one behind the audio tokenizer on a
Mac, CUDA's allocated and reserved figures on Windows. The checkpoint is loaded
once and reused, and the live figure stays flat no matter how many takes you
generate.

What used to climb was the allocators' pools of freed-but-retained blocks: over
five design takes on a Mac, MLX's cache went 0 → 1.9 GiB and torch's MPS pool
3.5 → 6.6 GiB, still rising. Anchored generation encodes a reference recording
on every chunk, which plain synthesis never does, so voice design leans on the
audio tokenizer far harder. CUDA's caching allocator behaves the same way.

Each backend bounds what it can at start-up and names the one pool it cannot,
which the engine then watches — `BREEZE_MLX_CACHE_LIMIT_GB` and
`BREEZE_MPS_CACHE_BUDGET_GB` on macOS, `BREEZE_CUDA_CACHE_BUDGET_GB` on Windows,
1 GiB each. Trimming happens after a document completes, with the generation gate
already released. Memory is flat from the first take onward and per-take time is
unchanged. Nothing is unloaded — only spare blocks are handed back.

## The four modes, and the CFG rule

Mode is inferred from what you send:

| Reference audio | Instruction | Mode | CFG allowed |
| :---: | :---: | :--- | :---: |
| – | – | `plain` | no |
| – | yes | `guided` (voice design) | **yes** |
| yes | – | `clone` (voice clone) | no |
| yes | yes | `edit` (voice direction) | **yes** |

`plain` and `clone` have no negative branch, so classifier-free guidance is
undefined and the upstream CLI *rejects* `--cfg-scale != 1`. The server clamps
it to `1.0` instead of erroring, so a cloning request that passes `cfg_scale: 4`
still succeeds — the value is simply ignored.

## Vocal events

Insert directly in the text. Only these are supported:

- English, parentheses: `(laugh)` `(cough)` `(clears throat)` `(sigh)`
- Chinese, square brackets: `[笑]` `[咳嗽]` `[清嗓子]` `[叹气]`

Anything else is read aloud literally, which is why `text_prep.py` filters the
model's output against this list.

## Text preparation

`prepare: true` sends the text to the configured language model first, which
adds punctuation, line breaks, and supported vocal events without changing the
words:

```
in  ok listen the deploy failed again i checked the logs and its the same null
    pointer as before im not happy about this

out Okay, listen.
    The deploy failed again.
    I checked the logs, and it's the same null pointer as before.
    I'm not happy about this. (sigh)
```

Which model does this is a setting — see
[Choosing a language model](#choosing-a-language-model). Prep failures are
non-fatal: the server falls back to the raw text and reports why in
`X-Breeze-Prep-Error`.

There are two paths through a language model in this project, and they exist for
different reasons. This one is a single request for a single answer, used from
the browser. The hotkey path uses `llm_stream.py`, which *streams*, because there
the first paragraph has to start being spoken while the model is still writing
the third. Both go through `llm_providers`, so the provider is configured once.

## Long text

`validate_and_chunk_text()` splits on sentence terminals, then on clause
boundaries for any sentence over `max_words`. Chunks are joined with a 220 ms
breath pause. Vocal-event tags contain no terminals, so they survive splitting.

Chunking is worth keeping — but for output *quality*, not memory. On a 96-word
single sentence, chunked rendering produced 39.2 s of audio while the same text
unchunked produced 78.6 s, i.e. the model drifted and padded badly without the
guardrail.

## Measured performance

Measured on an M4 Pro, 48 GB, INT8 checkpoint. The Windows figures will differ
with the card; the shape of them -- flat memory, sub-second first audio --
should not, since the pipeline above the runtime is the same code.


| Metric | Value |
| :--- | :--- |
| Model weights in memory | 3.46 GiB |
| Server steady RSS | ~4.0 GB |
| RSS growth during generation | **+0.02 GB peak** |
| Generation speed | 0.9× – 1.3× realtime |
| Time to first audio (streaming) | **0.19 s** |
| Added latency with `prepare: true` | ~1.5 s |
| Time to first token, Gemini 3.5 Flash Lite | ~0.6 – 1.0 s |

Concurrency: generation is serialized by a lock, since one runtime cannot run
two generations at once, on either platform. Requests queue and complete in order, and the event
loop stays free — `/health` answered in 11 ms with three generations queued.

## Troubleshooting

**The hotkey does nothing.** Run `python install_hotkeys.py --check`; it names
the cause. On macOS it is almost always Hammerspoon not running, or not holding
Accessibility permission (System Settings → Privacy & Security → Accessibility).
On Windows it is almost always another program owning the combination — Windows
gives a shortcut to whoever registered it first, silently — or the daemon not
running. Pick different keys on the **System speech** tab, and restart the
daemon.

**The server will not start: "no CUDA device is visible to PyTorch".** On a
machine that plainly has one, this means a CPU-only torch wheel got installed —
`pip install torch` does that on Windows unless the NVIDIA index is named.
Reinstall with `install.ps1`, or by hand:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If your driver is older, try `cu121`; `nvidia-smi` prints the highest CUDA
version it supports in its top-right corner.

**"The Qwen audio tokenizer requires the native SoX executable".** SoX is a
program, not a Python package. `brew install sox`, or
`winget install --id ChrisBagwell.SoX` and then open a new terminal so `PATH`
picks it up.

**Out of VRAM on Windows.** The INT8 checkpoint is about 5.2 GB of weights and
the BF16 one about 7 GB. Close other GPU applications; if you are on BF16,
`python download_model.py --variant torch-int8` saves nearly 2 GB.

**"is a torchao INT8 checkpoint, and torchao is not installed".** `pip install
torchao==0.17.0`, or use the BF16 checkpoint instead with
`python download_model.py --variant torch-bf16`. The pin matters: torchao 0.18+
needs torch 2.11, and the CUDA wheels the installer fetches are 2.9.

**Generation cannot keep up with playback on Windows.** Weight-only INT8 buys
disk and memory, not necessarily speed — on some hardware every linear
dequantizes before the matmul. Try `--variant torch-bf16` if you have the VRAM.

**Playback crackles.** See [Sample rates](#sample-rates-and-the-crackle-that-comes-from-getting-them-wrong).
The **System speech** tab shows the rate pair and underrun count live.

**Playback stutters and the underrun count climbs.** The engine is genuinely
behind. Raise the safety buffer on the **System speech** tab, or shorten
`max_words` so chunks come back sooner.

**"No saved voices yet".** Design or clone one first; the hotkey needs a voice
with a reference recording to anchor to.

**The model pass says unavailable.** `GET /v1/system-speech/config` reports which
provider was chosen and exactly why it could not answer — a missing key, a model
that refused, or no Google Cloud project. For Vertex specifically:

```bash
gcloud auth application-default login
gcloud config set project <your-project>
```

**A variation will not attach as a reference.** It is shorter than the ~10
seconds needed to clone from. Generate it on a longer script.

**The voice drifts across a long read.** Check the rotation settings — if the
variations are too unlike each other, the read will wander. Tag the ones that
belong together and set `tags` on the rotation so it only moves between those.

**Speech is cut off at the end.** The utterance was replaced by a newer one, or
stopped. `GET /v1/speak/status` reports the state and any error.

## Tests

All of them run offline in seconds -- a stub stands in for the engine, so none
needs the checkpoint, a GPU, or a network.

```bash
python test_voice_lock.py       # anchoring: which chunks get a reference, and why
python test_system_speech.py    # references, rotation, and the LLM-to-speech pipeline
python test_paragraph_skip.py   # paragraph seeking, pausing, and cleanup on abandon
python test_archive_audio.py    # sizing the archive, and exporting its audio safely
python test_torch_parity.py     # do the two speech backends compute the same thing?
python test_cross_platform.py   # shortcuts, providers, backends, .env parsing
```

None of them play audio on any device.

`test_torch_parity.py` is the one worth running after touching either runtime.
The failure mode of a hand-written port is not a crash -- it is a model that
runs, produces speech, and sounds subtly unlike the same model on the other
machine. It pushes identical random weights through both implementations and
compares the numbers, so a transposed rotary half or a mis-paired attention head
fails loudly instead of quietly changing the voice. It skips itself, rather than
failing, on a machine with only one backend installed.

`test_cross_platform.py` covers the seams where a Windows-only mistake would
otherwise sit unnoticed on a Mac: which shortcut strings are legal, which
provider a given `.env` resolves to, which checkpoint each backend expects, and
how a `.env` with CRLF line endings and quoted values parses. Most of that is
decisions rather than system calls, so it runs anywhere.

It also resolves every name the platform-specific modules load against the
scopes that could define it. Those branches never execute on the other machine,
so a name that does not exist in one sits there until someone tries Windows and
gets a `NameError` in the middle of a hotkey press. A linter catches this too —
use one if you have one; this is the stand-in for when you do not.

## Licence

Breeze TTS 2 weights are **research and non-commercial use only**
(BreezeBlue licence). The MLX runtime source is Apache-2.0; the PyTorch runtime
in `breeze-tts-torch/` is a port of it and carries the same licence.
