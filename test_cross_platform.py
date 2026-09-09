#!/usr/bin/env python3
"""Check the pieces that differ between macOS and Windows, from either one.

These are the seams where a Windows-only mistake would otherwise sit unnoticed
on a Mac until someone tried the other machine. Most of them are decisions, not
system calls -- which shortcut string is legal, which language-model provider a
configuration resolves to, which checkpoint a backend expects -- and decisions
can be checked anywhere.

The parts that genuinely need the other operating system (registering a global
hotkey, reading the Win32 clipboard, writing a Startup shortcut) are not tested
here. What is tested is everything leading up to them, so that when they run for
the first time on Windows they are given correct input.

    python test_cross_platform.py

Runs offline in under a second. Nothing here touches the model, the network, or
the saved configuration.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

PASSED = 0
FAILED: list[str] = []


def check(condition: bool, description: str) -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok    {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL  {description}")


def raises(exception_type, callable_, description: str) -> None:
    try:
        callable_()
    except exception_type:
        check(True, description)
    except Exception as exc:  # noqa: BLE001
        check(False, f"{description} (raised {type(exc).__name__}: {exc})")
    else:
        check(False, f"{description} (nothing raised)")


@contextmanager
def environment(**values: str | None):
    """Set environment variables for one block, and put them back after."""
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------------------
def test_bindings() -> None:
    """The shortcut notation both hotkey hosts have to agree on."""
    print("\nshortcuts")
    import hotkeys
    from hotkeys.bindings import InvalidBinding

    check(
        hotkeys.canonical("Alt+Ctrl+S") == "ctrl+alt+s",
        "modifiers are reordered and lowercased, so two spellings compare equal",
    )
    check(
        hotkeys.canonical("Command+shift+Left") == "shift+cmd+left",
        "cmd and command are the same key role",
    )
    check(
        hotkeys.canonical("win+s") == hotkeys.canonical("cmd+s"),
        "so is the Windows key: a config written on one machine loads on the other",
    )

    raises(InvalidBinding, lambda: hotkeys.canonical("s"),
           "a bare key is refused -- it would fire while typing")
    raises(InvalidBinding, lambda: hotkeys.canonical("ctrl+alt+ss"),
           "a key name no host would recognise is refused")
    raises(InvalidBinding, lambda: hotkeys.canonical("hyper+s"),
           "an unknown modifier is refused")
    raises(
        InvalidBinding,
        lambda: hotkeys.validate(
            {**hotkeys.DEFAULTS, "toggle": hotkeys.DEFAULTS["stop"]}
        ),
        "two actions on one combination is refused, not resolved by order",
    )

    validated = hotkeys.validate({})
    check(set(validated) == set(hotkeys.DEFAULTS), "defaults validate as a complete set")
    check(
        all(action in validated for action, _, _ in hotkeys.ACTIONS),
        "every documented action has a binding",
    )

    # The daemon's translation into pynput's notation. Pure string work, so it
    # is checkable without pynput installed.
    from hotkeys.daemon import ENDPOINTS, to_pynput

    check(to_pynput("ctrl+alt+s") == "<ctrl>+<alt>+s", "letters stay bare for pynput")
    check(to_pynput("ctrl+alt+right") == "<ctrl>+<alt>+<right>", "named keys are bracketed")
    check(to_pynput("ctrl+alt+esc") == "<ctrl>+<alt>+<escape>", "esc is spelled out for pynput")
    check(
        set(ENDPOINTS) == set(hotkeys.DEFAULTS),
        "every action the daemon can bind has somewhere to post",
    )


def test_providers() -> None:
    """Which language model a given .env resolves to."""
    print("\nlanguage-model providers")
    import llm_providers
    from llm_providers.base import Provider

    check(len(llm_providers.REGISTRY) >= 4, "vertex, gemini and both OpenRouter tiers")
    check(
        all(isinstance(p, Provider) for p in llm_providers.REGISTRY.values()),
        "every entry implements the interface, so adding one cannot half-work",
    )
    check(
        all(p.name and p.label for p in llm_providers.REGISTRY.values()),
        "every provider names itself for the settings UI",
    )

    keys = {"GEMINI_API_KEY": None, "GOOGLE_API_KEY": None,
            "OPENROUTER_API_KEY": None, "BREEZE_LLM_PROVIDER": None}

    with environment(**{**keys, "BREEZE_LLM_PROVIDER": "openrouter_paid"}):
        check(
            llm_providers.resolve_name() == "openrouter_paid",
            "an explicit provider wins even with no key -- the error then names the key",
        )
    with environment(**keys):
        raises(
            llm_providers.ProviderUnavailable,
            lambda: llm_providers.resolve_name("nonesuch"),
            "an unknown provider name is refused with the valid list",
        )
    with environment(**{**keys, "GEMINI_API_KEY": "x"}):
        check(
            llm_providers.resolve_name() == "gemini",
            "auto picks the provider whose key is actually present",
        )
    with environment(**{**keys, "OPENROUTER_API_KEY": "x"}):
        check(
            llm_providers.resolve_name() == "openrouter_free",
            "...whichever one that is",
        )
    with environment(**keys):
        check(
            llm_providers.resolve_name() == "vertex",
            "with no key at all it falls to Vertex, which needs none",
        )
    with environment(**{**keys, "GEMINI_API_KEY": "x", "OPENROUTER_API_KEY": "y"}):
        check(
            llm_providers.resolve_name() == "gemini",
            "with several keys the order in AUTO_ORDER decides, not dict order",
        )

    rows = llm_providers.describe_all()
    check(len(rows) == len(llm_providers.REGISTRY), "the settings UI is served the whole registry")
    check(
        all("default_model" in row and "configured" in row for row in rows),
        "each row says what it would use and whether it could",
    )

    # The reusable OpenRouter module: no project imports, and no network here.
    import llm_providers.openrouter as openrouter

    check(
        openrouter.FREE_ROUTER == "openrouter/free",
        "the zero-discovery fallback slug is present",
    )
    music = openrouter.Model(
        id="x/lyria", name="music", context_length=10**6,
        prompt_price=0.0, completion_price=0.0,
        input_modalities=("text",), output_modalities=("text", "audio"),
    )
    chat = openrouter.Model(
        id="x/instruct", name="chat", context_length=8192,
        prompt_price=0.0, completion_price=0.0,
    )
    check(music.is_free and not music.is_text_chat,
          "a free music model is free but is not a chat model")
    check(chat.is_free and chat.is_text_chat, "a free text model is both")
    with environment(OPENROUTER_API_KEY=None):
        raises(
            openrouter.OpenRouterError,
            lambda: openrouter.OpenRouterClient(),
            "a missing key is refused before any request is made",
        )


def test_backends() -> None:
    """Which speech runtime this machine would load, and what it expects."""
    print("\nspeech backends")
    import tts_backends

    survey = {row["name"]: row for row in tts_backends.survey()}
    check(set(survey) == {"mlx", "torch"}, "both backends are reported, available or not")
    check(
        all(row["available"] or row.get("reason") for row in survey.values()),
        "an unavailable backend says why, rather than merely being absent",
    )

    from tts_backends import mlx_backend, torch_backend

    for backend in (mlx_backend, torch_backend):
        for attribute in (
            "NAME", "DEFAULT_MODEL_PATH", "GROWTH_KEY", "GROWTH_BUDGET_ENV",
            "unavailable_reason", "describe", "build", "memory_stats", "trim_memory",
        ):
            check(
                hasattr(backend, attribute),
                f"{backend.NAME} backend provides {attribute}",
            )
    check(
        mlx_backend.DEFAULT_MODEL_PATH != torch_backend.DEFAULT_MODEL_PATH,
        "the two checkpoints live in different directories, so both can coexist",
    )

    with environment(BREEZE_BACKEND="nonesuch"):
        raises(
            tts_backends.BackendUnavailable,
            tts_backends.load_backend,
            "an unknown BREEZE_BACKEND is refused rather than silently ignored",
        )

    # download_model.py has to agree with the backends about the directories.
    # It is not a one-to-one map: the CUDA backend accepts either an INT8 or a
    # BF16 checkpoint, so what has to hold is that every variant lands somewhere
    # its backend looks, and that each backend's default is downloadable.
    import download_model

    modules = {"mlx": mlx_backend, "torch": torch_backend}
    for name, source in download_model.SOURCES.items():
        backend = modules[str(source["backend"])]
        looks_in = {path.lstrip("./") for path in backend.CANDIDATE_MODEL_PATHS}
        check(
            str(source["dest"]) in looks_in,
            f"the downloader puts {name} where the {backend.NAME} backend looks",
        )
        check(bool(source["required"]), f"the {name} download is checked for completeness")

    for backend_name, variant in download_model.DEFAULT_VARIANT.items():
        check(
            download_model.SOURCES[variant]["backend"] == backend_name,
            f"the default {backend_name} variant is one of its own",
        )
        check(
            str(download_model.SOURCES[variant]["dest"])
            == modules[backend_name].DEFAULT_MODEL_PATH.lstrip("./"),
            f"and it lands where {backend_name} looks first",
        )

    check(
        torch_backend.resolve_model_path().lstrip("./")
        in {str(source["dest"]) for source in download_model.SOURCES.values()},
        "the CUDA backend resolves to a checkpoint the downloader can fetch",
    )


def test_platform_support() -> None:
    """Paths, .env parsing, and the reporting the UI relies on."""
    print("\nplatform support")
    import platform_support

    described = platform_support.describe()
    check(
        {"system", "release", "machine", "python", "hotkey_host"} <= set(described),
        "the UI is told which platform and hotkey host it is talking to",
    )
    check(
        described["hotkey_host"] in {"hammerspoon", "python"},
        "the hotkey host is one this project actually ships",
    )
    check(
        sum([platform_support.IS_MACOS, platform_support.IS_WINDOWS,
             platform_support.IS_LINUX]) <= 1,
        "the platform flags are mutually exclusive",
    )
    check(
        Path(platform_support.python_executable()).exists(),
        "the interpreter a login item would relaunch actually exists",
    )

    # The fallback .env reader, used when python-dotenv is not installed. It has
    # to cope with what a real file contains: comments, blanks, quotes, CRLF,
    # `export`, and a value with an = in it.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ".env"
        path.write_bytes(
            b"# a comment\r\n"
            b"\r\n"
            b'BREEZE_TEST_QUOTED="quoted value"\r\n'
            b"export BREEZE_TEST_EXPORTED=exported\r\n"
            b"BREEZE_TEST_EQUALS=a=b=c\r\n"
            b"BREEZE_TEST_ALREADY=from-file\r\n"
        )
        with environment(
            BREEZE_TEST_QUOTED=None, BREEZE_TEST_EXPORTED=None,
            BREEZE_TEST_EQUALS=None, BREEZE_TEST_ALREADY="from-shell",
        ):
            platform_support._load_env_minimal(path, override=False)
            check(os.environ["BREEZE_TEST_QUOTED"] == "quoted value",
                  "quotes are stripped from a value")
            check(os.environ["BREEZE_TEST_EXPORTED"] == "exported",
                  "an `export` prefix is ignored")
            check(os.environ["BREEZE_TEST_EQUALS"] == "a=b=c",
                  "only the first = separates name from value")
            check(os.environ["BREEZE_TEST_ALREADY"] == "from-shell",
                  "the shell wins over the file, so a one-off override works")

    check(
        platform_support.ENV_PATH.parent == PROJECT,
        ".env is read from the project directory, not the working directory",
    )


def test_config_round_trip() -> None:
    """Shortcuts survive being saved and loaded, without touching the real file."""
    print("\nconfiguration")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        with environment(BREEZE_STATE_DIR=directory):
            # Imported inside the override: the module resolves its path once.
            for name in ("speech_config",):
                sys.modules.pop(name, None)
            import speech_config

            check(
                "hotkeys" in speech_config.DEFAULTS,
                "the shortcuts are part of the same config as everything else",
            )
            saved = speech_config.save({"hotkeys": {"toggle": "Ctrl+Shift+9"}})
            check(saved["hotkeys"]["toggle"] == "ctrl+shift+9",
                  "a saved shortcut is normalized on the way in")
            check(saved["hotkeys"]["stop"] == "ctrl+alt+x",
                  "and the ones not mentioned are left alone")
            check(speech_config.load()["hotkeys"]["toggle"] == "ctrl+shift+9",
                  "it survives a reload")
            raises(
                ValueError,
                lambda: speech_config.save({"hotkeys": {"toggle": "nonsense"}}),
                "an unusable shortcut is refused before it reaches disk",
            )
            check(speech_config.load()["hotkeys"]["toggle"] == "ctrl+shift+9",
                  "and the refused write changed nothing")

    # Put the module back the way the rest of the process expects it.
    sys.modules.pop("speech_config", None)



# ---------------------------------------------------------------------------
# Undefined names in code this machine never runs
#
# The Windows branches -- the Win32 clipboard, the COM device enumerator, the
# Startup shortcut -- are never executed on a Mac, so a name that does not exist
# in one of them sits there silently until someone tries the other machine, and
# then fails as a NameError in the middle of a hotkey press. That is precisely
# the bug this port is prone to, and it is the one that already happened once.
#
# A linter catches this, and if one is installed it should be used. This is a
# small stand-in so the check exists unconditionally: it resolves every name a
# scope *loads* against its own bindings, its enclosing scopes and the builtins.
#
# It errs deliberately towards silence. Names bound anywhere in a scope count as
# visible everywhere in it, so use-before-assignment is not reported; class
# bodies are treated as ordinary enclosing scopes; comprehension targets are
# folded into the scope around them. All of those would need real flow analysis
# to get right, and a false alarm here would train people to ignore it.
# ---------------------------------------------------------------------------
import ast
import builtins

PLATFORM_MODULES = (
    "platform_support.py",
    "audio_out.py",
    "hotkeys/daemon.py",
    "hotkeys/bindings.py",
    "hotkeys/install_windows.py",
    "hotkeys/install_macos.py",
    "tts_backends/__init__.py",
    "tts_backends/torch_backend.py",
    "tts_backends/mlx_backend.py",
    "llm_providers/openrouter.py",
    "download_model.py",
)

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _own_nodes(node: ast.AST):
    """Every node in this scope, stopping at the boundary of a nested one."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if not isinstance(child, _SCOPES):
            stack.extend(ast.iter_child_nodes(child))


def _bound_names(node: ast.AST) -> set[str]:
    """Everything this scope binds, without descending into nested scopes."""
    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = node.args
        for group in (arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
            names.update(argument.arg for argument in group)
        for extra in (arguments.vararg, arguments.kwarg):
            if extra is not None:
                names.add(extra.arg)
    for child in _own_nodes(node):
        if isinstance(child, _SCOPES):
            # The nested scope's own name is bound here; its body is not.
            names.add(getattr(child, "name", ""))
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
        elif isinstance(child, ast.comprehension):
            for target in ast.walk(child.target):
                if isinstance(target, ast.Name):
                    names.add(target.id)
    names.discard("")
    return names


def _check_scope(node, enclosing: set[str], problems: list[str], where: str) -> None:
    visible = enclosing | _bound_names(node)
    for child in _own_nodes(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in visible:
                problems.append(f"line {child.lineno}: {child.id} in {where}")
    for child in _own_nodes(node):
        if isinstance(child, _SCOPES):
            name = getattr(child, "name", "<lambda>")
            _check_scope(child, visible, problems, f"{where}.{name}" if where else name)


def undefined_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    known = set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__spec__", "__package__", "__builtins__",
    }
    problems: list[str] = []
    _check_scope(tree, known, problems, "")
    return sorted(set(problems))


def test_no_undefined_names() -> None:
    print("\nnames in code this machine never runs")
    for relative in PLATFORM_MODULES:
        path = PROJECT / relative
        problems = undefined_names(path)
        check(not problems, f"{relative} references only names that exist")
        for problem in problems:
            print(f"        {problem}")

    # And the check itself is worth checking: a stand-in for a linter that never
    # reports anything is worse than no check at all.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        planted = Path(directory) / "planted.py"
        planted.write_text(
            "import os\n"
            "def outer():\n"
            "    import sys\n"
            "    def inner():\n"
            "        return sys.path, os.sep, typo_that_does_not_exist\n"
            "    return inner\n"
        )
        found = undefined_names(planted)
        check(
            any("typo_that_does_not_exist" in problem for problem in found),
            "the check finds a name that exists nowhere",
        )
        check(
            not any(("sys" in p or "os" in p) for p in found),
            "...without flagging imports from an enclosing scope",
        )



def test_checkpoint_formats() -> None:
    """The two shard formats, both loaded on whichever machine this is.

    Windows reads either the original safetensors shards or the pickle shards a
    torchao INT8 conversion ships as, and picks the second by default. Neither
    path runs on a Mac in normal use, so both are exercised here on a model
    small enough to build in memory: a checkpoint is written in the *source*
    naming the real ones use, loaded back, and compared tensor for tensor.

    What this cannot check is the quantized weights themselves -- making one
    needs torchao and a GPU. What it does check is everything around them: the
    index, the shard reader, the name mapping, the codebook-head split, and the
    meta-device construction that the loading depends on.
    """
    print("\ncheckpoint formats")
    try:
        import torch
    except ImportError:
        check(True, "skipped: torch is not installed")
        return

    import json
    import tempfile

    sys.path.insert(0, str(PROJECT / "breeze-tts-torch"))
    sys.path.insert(0, str(PROJECT / "breeze-tts-mlx"))
    from breeze_tts_mlx.config import BreezeMLXConfig
    from breeze_tts_torch.model import BreezeTorchModel, find_index

    # The shrunken configuration the parity test already defines: the real
    # one's shape, small enough to build twice in memory.
    from test_torch_parity import BASE_CONFIG

    torch.set_grad_enabled(False)

    def to_source_names(state: dict) -> dict:
        """Our parameter names, back to the ones a checkpoint actually uses.

        The inverse of ``map_source_tensor``. Written out rather than derived,
        so that a change to the mapping fails this test instead of being
        mirrored into it automatically.
        """
        heads: dict[int, torch.Tensor] = {}
        source: dict[str, torch.Tensor] = {}
        for name, tensor in state.items():
            if name.startswith("depth_decoder.codebooks_head.heads."):
                index = int(name.split(".")[3])
                heads[index] = tensor.T.contiguous()
            elif name == "depth_decoder.input_projection.weight":
                source["depth_decoder.model.inputs_embeds_projector.weight"] = tensor
            elif name.startswith("depth_decoder."):
                source["depth_decoder.model." + name[len("depth_decoder."):]] = tensor
            elif name.startswith("backbone."):
                source["backbone_model." + name[len("backbone."):]] = tensor
            elif name == "audio_embedding.embedding.weight":
                source["depth_decoder.model.embed_tokens.weight"] = tensor
            elif name == "text_encoder.embed_tokens.embedding.weight":
                source["text_encoder.embed_tokens.weight"] = tensor
            else:
                source[name] = tensor
        source["depth_decoder.codebooks_head.weight"] = torch.stack(
            [heads[index] for index in sorted(heads)]
        )
        return source

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(BASE_CONFIG))
        reference = BreezeTorchModel(BreezeMLXConfig.from_file(config_path))
        reference.eval()
        original = {
            name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            for name, tensor in reference.state_dict().items()
        }
        source = to_source_names(original)

        # Split across two shards, so the per-shard loop is exercised rather
        # than a single-file shortcut.
        names = sorted(source)
        halves = (names[: len(names) // 2], names[len(names) // 2 :])

        for kind, index_name, shard_pattern, writer in (
            (
                "safetensors",
                "model.safetensors.index.json",
                "model-{n:05d}-of-00002.safetensors",
                None,
            ),
            (
                "pickle",
                "pytorch_model.bin.index.json",
                "pytorch_model-{n:05d}-of-00002.bin",
                torch.save,
            ),
        ):
            checkpoint = root / kind
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps(BASE_CONFIG))
            weight_map = {}
            for number, half in enumerate(halves, start=1):
                shard_name = shard_pattern.format(n=number)
                payload = {name: source[name] for name in half}
                if writer is None:
                    from safetensors.torch import save_file

                    save_file(
                        {k: v.contiguous() for k, v in payload.items()},
                        str(checkpoint / shard_name),
                    )
                else:
                    writer(payload, checkpoint / shard_name)
                weight_map.update({name: shard_name for name in half})
            (checkpoint / index_name).write_text(json.dumps({"weight_map": weight_map}))

            _path, detected = find_index(checkpoint)
            check(detected == kind, f"a {kind} checkpoint is recognised as one")

            loaded = BreezeTorchModel.from_checkpoint(
                checkpoint, device="cpu", dtype=torch.bfloat16
            )
            state = loaded.state_dict()
            check(
                set(state) == set(original),
                f"{kind}: every parameter is filled, and no extra ones appear",
            )
            worst = max(
                float((state[name].float() - original[name].float()).abs().max())
                for name in original
            )
            check(worst == 0.0, f"{kind}: every weight round-trips exactly")
            check(
                not any(t.is_meta for t in state.values()),
                f"{kind}: nothing is left on the meta device",
            )
            check(
                all(not p.requires_grad for p in loaded.parameters()),
                f"{kind}: loaded for inference, not training",
            )

        # A directory with neither index is refused by name, not by a later
        # failure inside a matmul.
        empty = root / "empty"
        empty.mkdir()
        (empty / "config.json").write_text(json.dumps(BASE_CONFIG))
        raises(
            FileNotFoundError,
            lambda: BreezeTorchModel.from_checkpoint(empty, device="cpu"),
            "a directory that is not a checkpoint is refused with what is missing",
        )



def test_requirements() -> None:
    """One requirements file, and what it resolves to on each machine.

    The markers are the whole reason a single file works, and they are exactly
    the kind of thing that is wrong in a way nobody notices: a marker that never
    matches installs nothing and says nothing, and one that always matches tries
    to put MLX on a PC. So both sides are evaluated here rather than trusted.
    """
    print("\nrequirements")
    try:
        from packaging.requirements import Requirement
    except ImportError:
        check(True, "skipped: packaging is not installed")
        return

    path = PROJECT / "requirements.txt"
    check(path.is_file(), "there is one requirements.txt, at the project root")
    check(
        not (PROJECT / "requirements").exists(),
        "...and no per-platform directory beside it",
    )

    entries = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            # A -r or --index-url line. The second would silently change where
            # every package in the file comes from, which is not something to
            # leave to a review.
            check(False, f"line {number} is a pip option, not a requirement: {line}")
            continue
        try:
            entries.append(Requirement(line))
        except Exception as exc:  # noqa: BLE001
            check(False, f"line {number} does not parse: {line} ({exc})")
    check(bool(entries), "the file parses as requirements")

    # A complete marker environment for each platform. Every key pip can test
    # has to be present, or evaluating a marker raises rather than returning
    # False -- which would look like "this package is skipped here".
    def machine(**overrides: str) -> dict[str, str]:
        base = {
            "implementation_name": "cpython",
            "implementation_version": "3.12.0",
            "os_name": "posix",
            "platform_machine": "arm64",
            "platform_python_implementation": "CPython",
            "platform_release": "",
            "platform_system": "Darwin",
            "platform_version": "",
            "python_full_version": "3.12.0",
            "python_version": "3.12",
            "sys_platform": "darwin",
            "extra": "",
        }
        base.update(overrides)
        return base

    machines = {
        "macos": machine(),
        "windows": machine(
            os_name="nt", platform_machine="AMD64", platform_system="Windows",
            sys_platform="win32",
        ),
        "linux": machine(
            platform_machine="x86_64", platform_system="Linux", sys_platform="linux",
        ),
    }
    selected = {
        name: {
            entry.name.lower().replace("_", "-")
            for entry in entries
            if entry.marker is None or entry.marker.evaluate(env)
        }
        for name, env in machines.items()
    }

    # What every machine needs, whichever backend it ends up running.
    shared = {
        "fastapi", "uvicorn", "python-multipart", "numpy", "soundfile", "soxr",
        "sounddevice", "transformers", "safetensors", "qwen-tts",
        "huggingface-hub", "torch", "torchaudio", "pynput", "python-dotenv",
        "requests", "google-genai", "psutil",
    }
    for name, packages in selected.items():
        missing = shared - packages
        check(not missing, f"{name} installs everything shared" + (f" (missing {sorted(missing)})" if missing else ""))

    check("mlx" in selected["macos"], "macOS installs MLX")
    check("mlx" not in selected["windows"], "Windows does not try to install MLX")
    check("mlx" not in selected["linux"], "nor does Linux, which has no Metal")
    check("torchao" in selected["windows"], "Windows installs torchao, for the INT8 checkpoint")
    check("torchao" not in selected["macos"], "macOS does not, since MLX does not use it")

    # Every third-party module the project imports has to be installable from
    # this file. Names differ from imports often enough that a missing one is a
    # plausible mistake, so the mapping is written out.
    distributions = {
        "dotenv": "python-dotenv", "google": "google-genai", "qwen_tts": "qwen-tts",
        "huggingface_hub": "huggingface-hub", "starlette": "fastapi",
    }
    imports = {
        "dotenv", "fastapi", "google", "huggingface_hub", "mlx", "numpy", "psutil",
        "pynput", "qwen_tts", "requests", "safetensors", "sounddevice", "soundfile",
        "soxr", "starlette", "torch", "torchao", "transformers", "uvicorn",
    }
    everywhere = set().union(*selected.values())
    for module in sorted(imports):
        distribution = distributions.get(module, module).lower().replace("_", "-")
        check(
            distribution in everywhere,
            f"`import {module}` is covered by {distribution}",
        )



def test_entry_points_read_env() -> None:
    """Anything documented as a .env setting has to be read by whoever uses it.

    ``.env`` is per-machine configuration, and a script that looks up a setting
    without loading the file first does not fail -- it quietly uses the default
    and does exactly the wrong thing. That happened once already, to the
    downloader, and it is invisible on a machine where the default is right.
    """
    print("\nentry points and .env")
    import ast

    # Modules a user runs directly, and which resolve at least one BREEZE_*
    # setting. Library modules are excluded: they are imported by these, and by
    # then the file has been read.
    entry_points = ("breeze_server.py", "download_model.py", "text_prep.py")

    for relative in entry_points:
        source = (PROJECT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        loads_env = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_env"
            for node in ast.walk(tree)
        )
        check(loads_env, f"{relative} loads .env before it reads a setting")

    # And every setting .env.example documents is one something actually reads,
    # so the file cannot drift into promising options that do nothing.
    documented = set()
    for raw in (PROJECT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip().lstrip("#").strip()
        if "=" in line and not line.startswith(" "):
            name = line.split("=", 1)[0].strip()
            if name.isupper() and name.replace("_", "").isalnum():
                documented.add(name)
    check(bool(documented), ".env.example documents some settings")

    haystack = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PROJECT.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    )
    unused = sorted(name for name in documented if name not in haystack)
    check(not unused, f"every documented setting is read somewhere" + (f" (orphans: {unused})" if unused else ""))



# The one voice that ships, so a fresh clone can speak before the user has
# recorded anything. Named here rather than discovered, because "whatever
# happens to be in voices/" is exactly the thing this test exists to prevent.
BUNDLED_VOICE = "voice_132150d40e9e455b97362ad6"


def test_bundled_voice() -> None:
    """The shipped voice: exactly one, complete, and readable on this platform.

    Two different failures are being guarded here.

    *Too much shipping.* voices/ holds recordings of real people. Exactly one is
    meant to be in the repository, and the .gitignore that arranges that is
    fiddly -- a negation under an excluded directory never matches, and one
    placed above `*.wav` is overruled by it. Either mistake is silent: the
    profile commits and the audio does not, or seven other people's voices go
    along for the ride.

    *Audio that does not decode.* The reference is MPEG-in-WAV, which needs the
    MPEG support in libsndfile. Every recent `soundfile` wheel has it, on both
    platforms -- but if a build ever does not, this is where it should fail,
    rather than in the middle of someone's first attempt to hear anything.
    """
    print("\nbundled voice")
    import json
    import subprocess

    directory = PROJECT / "voices" / BUNDLED_VOICE
    check(directory.is_dir(), f"{BUNDLED_VOICE} is present")
    if not directory.is_dir():
        return

    profile_path = directory / "profile.json"
    check(profile_path.is_file(), "it has a profile.json")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    check(profile.get("voice_id") == BUNDLED_VOICE, "whose voice_id matches its directory")
    references = profile.get("references") or []
    check(bool(references), "and at least one reference")

    for reference in references:
        audio = directory / str(reference.get("file"))
        check(audio.is_file(), f"reference {reference.get('reference_id')} has its audio file")

    # Decoded the way encode_prompt_audio decodes it, not merely opened.
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        check(True, "skipped: soundfile is not installed")
        return

    audio_path = directory / str(profile["reference"]["file"])
    try:
        samples, rate = sf.read(audio_path, always_2d=True, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        check(False, f"the reference decodes ({exc})")
        return
    mono = np.mean(samples, axis=1)
    seconds = len(mono) / rate
    check(bool(np.isfinite(mono).all()), "the reference decodes to finite samples")
    check(float(np.abs(mono).max()) > 0.01, "...that are not silence")

    import voice_store

    check(
        seconds >= voice_store.REFERENCE_MIN_SECONDS,
        f"and it is long enough to clone from ({seconds:.1f}s, "
        f"minimum {voice_store.REFERENCE_MIN_SECONDS:.0f}s)",
    )

    # What the repository actually carries, asked of git rather than of the
    # working tree -- the untracked voices sitting beside it are the whole point.
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "voices/"],
            cwd=PROJECT, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        check(True, "skipped: git is not available to check what is committed")
        return
    if tracked.returncode != 0:
        check(True, "skipped: not a git checkout")
        return
    committed = [line for line in tracked.stdout.splitlines() if line.strip()]
    identifiers = {line.split("/")[1] for line in committed if line.startswith("voices/")}
    check(
        identifiers == {BUNDLED_VOICE},
        f"exactly one voice is committed" + (f" (found {sorted(identifiers)})" if identifiers != {BUNDLED_VOICE} else ""),
    )
    check(
        any(line.endswith(".wav") for line in committed),
        "...including its audio, which the *.wav rule would otherwise have eaten",
    )



def test_nothing_private_is_committed() -> None:
    """The repository must not carry anything the local machine generated.

    This is the same question ``check_private_files.py`` answers by hand, run
    automatically so that a change to .gitignore cannot quietly start publishing
    the archive. It asks git, not the filesystem: a path is a problem if it has
    ever been committed, because removing it later leaves the blob in the
    history and in every clone.
    """
    print("\nprivacy")
    import subprocess

    import check_private_files as auditor

    try:
        status, _ = auditor.run("rev-parse", "--git-dir")
    except (OSError, subprocess.SubprocessError):
        check(True, "skipped: git is not available")
        return
    if status != 0:
        check(True, "skipped: not a git checkout")
        return

    _status, output = auditor.run("ls-files")
    tracked = {line for line in output.splitlines() if line.strip()}
    check(bool(tracked), "git reports tracked files")
    offenders = auditor.classify(tracked)
    check(
        not offenders,
        "nothing private is tracked" + (f" (found {offenders})" if offenders else ""),
    )

    _status, output = auditor.run("log", "--pretty=format:", "--name-only", "HEAD")
    history = {line for line in output.splitlines() if line.strip()}
    offenders = auditor.classify(history)
    check(
        not offenders,
        "nor anywhere in this branch's history"
        + (f" (found {offenders})" if offenders else ""),
    )

    # The exceptions have to actually work, or the rule is doing nothing and the
    # bundled voice is not really shipping.
    check(
        auditor.is_allowed(f"voices/{auditor.BUNDLED_VOICE}/reference.wav"),
        "the bundled voice is allowed through",
    )
    check(
        auditor.is_allowed(f"voices/{auditor.BUNDLED_VOICE}"),
        "...as a directory entry too, which is how rev-list reports it",
    )
    check(
        not auditor.is_allowed("voices/voice_somebodyelse/reference.wav"),
        "and no other voice is",
    )
    check(
        auditor.is_allowed(".env.example") and not auditor.is_allowed(".env"),
        ".env.example ships; .env does not",
    )
    for path in ("state/archive/2026-01-01/utt_x/audio.wav", "outputs/take.wav"):
        check(bool(auditor.classify({path})), f"{path} would be caught")


def main() -> int:
    print("cross-platform checks")
    test_bindings()
    test_providers()
    test_backends()
    test_platform_support()
    test_config_round_trip()
    test_no_undefined_names()
    test_checkpoint_formats()
    test_requirements()
    test_entry_points_read_env()
    test_bundled_voice()
    test_nothing_private_is_committed()

    print()
    if FAILED:
        print(f"{len(FAILED)} failed, {PASSED} passed:")
        for description in FAILED:
            print(f"  - {description}")
        return 1
    print(f"{PASSED} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
