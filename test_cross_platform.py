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
    import download_model

    for name, source in download_model.SOURCES.items():
        expected = {"mlx": mlx_backend, "torch": torch_backend}[name].DEFAULT_MODEL_PATH
        check(
            expected.lstrip("./") == str(source["dest"]),
            f"the downloader puts the {name} checkpoint where the backend looks",
        )
        check(bool(source["required"]), f"the {name} download is checked for completeness")


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


def main() -> int:
    print("cross-platform checks")
    test_bindings()
    test_providers()
    test_backends()
    test_platform_support()
    test_config_round_trip()
    test_no_undefined_names()

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
