#!/usr/bin/env python3
"""Audit what this repository would publish, and what stays on this machine.

    python check_private_files.py            # this branch's whole history
    python check_private_files.py --all      # every branch, and the reflog
    python check_private_files.py --remote   # ...and what GitHub actually holds

Speaking a document leaves traces: the text that was read, the audio that was
produced, the recording a voice was cloned from, the API keys that paid for the
language-model pass. All of it is meant to stay on the machine that made it, and
all of it is arranged to by .gitignore -- but "arranged to" is not the same as
"checked", and the failure is silent. A file that should have been ignored and
was not looks exactly like a file that was meant to be committed.

So this looks at what git *actually has*, not at what .gitignore says. A path is
reported if it has ever appeared in the history being examined, even if a later
commit removed it: git keeps the blob, and anyone who clones gets it.

Exit status is 0 when nothing private is found, 1 when something is.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

# The one voice that ships on purpose. Everything else under voices/ is a
# recording of a real person and belongs only on the machine that made it.
BUNDLED_VOICE = "voice_132150d40e9e455b97362ad6"

# What must never be committed, and why -- the reason is printed, because
# "state/ is private" is not useful to someone deciding whether to worry.
PRIVATE = (
    (
        "state/archive/",
        "every utterance ever spoken: the text you copied, what the language "
        "model made of it, and the audio",
    ),
    ("state/", "system-speech settings and the utterance archive"),
    ("outputs/", "generated audio from development and testing"),
    (".env", "API keys, and this machine's model and project settings"),
    ("voices/", "voice profiles and the recordings they clone from"),
)

# Anything matching these is allowed even though it sits under a private root.
ALLOWED = (
    ".env.example",
    f"voices/{BUNDLED_VOICE}/",
)


def run(*arguments: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *arguments], cwd=PROJECT, capture_output=True, text=True, check=False
    )
    return result.returncode, result.stdout


def is_allowed(path: str) -> bool:
    """Whether a path is one of the deliberate exceptions.

    Directory entries arrive without a trailing slash from ``rev-list
    --objects``, and with one from nowhere at all, so both spellings of a rule
    have to match -- otherwise the bundled voice's own directory is reported as
    a leak of itself.
    """
    for rule in ALLOWED:
        base = rule.rstrip("/")
        if path == base or path.startswith(base + "/"):
            return True
    return False


def classify(paths: set[str]) -> dict[str, list[str]]:
    """Group the offending paths by the rule that should have excluded them."""
    found: dict[str, list[str]] = {}
    for path in sorted(paths):
        if is_allowed(path):
            continue
        for prefix, _reason in PRIVATE:
            if path == prefix or path.startswith(prefix):
                found.setdefault(prefix, []).append(path)
                break
    return found


def reason_for(prefix: str) -> str:
    return next(reason for rule, reason in PRIVATE if rule == prefix)


def report(title: str, paths: set[str]) -> bool:
    """Print one section. True when it is clean."""
    offenders = classify(paths)
    if not offenders:
        print(f"  clean   {title}")
        return True
    print(f"  FOUND   {title}")
    for prefix, files in offenders.items():
        print(f"            {prefix}  -- {reason_for(prefix)}")
        for path in files[:8]:
            print(f"              {path}")
        if len(files) > 8:
            print(f"              ... and {len(files) - 8} more")
    return False


def local_stores() -> None:
    """Where the private data lives on this machine, and how much of it."""
    print("\nOn this machine, outside git")
    print("---------------------------")
    entries = (
        ("state/archive/", "spoken history: text, audio and metadata per utterance"),
        ("state/system_speech.json", "the System speech settings"),
        ("voices/", "the voice library"),
        ("outputs/", "generated audio from development"),
        (".env", "API keys and machine settings"),
    )
    for relative, description in entries:
        path = PROJECT / relative
        if not path.exists():
            print(f"  --      {relative:<26} not present")
            continue
        if path.is_dir():
            files = [child for child in path.rglob("*") if child.is_file()]
            size = sum(child.stat().st_size for child in files)
            detail = f"{len(files)} file(s), {size / 1048576:.1f} MB"
        else:
            detail = f"{path.stat().st_size} bytes"
        print(f"  present {relative:<26} {detail}")
        print(f"          {description}")

    index = PROJECT / "state" / "archive" / "index.jsonl"
    if index.is_file():
        lines = [line for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"\n  The archive holds {len(lines)} utterance(s). To read what is in it:")
        print("      python -c \"import archive, json; [print(e['created_at'], e['input_preview'][:60]) for e in archive.recent(500)]\"")
        print("  To erase it: delete state/archive/, or use the System speech tab's")
        print("  'Clear history'. Turn it off entirely with archive.enabled = false there.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="every branch and the reflog, not just this one")
    parser.add_argument("--remote", action="store_true",
                        help="also ask GitHub what it actually holds (needs gh)")
    args = parser.parse_args(argv)

    status, _ = run("rev-parse", "--git-dir")
    if status != 0:
        print("Not a git repository.", file=sys.stderr)
        return 1

    print("What this repository would publish")
    print("==================================")
    published_clean = True
    local_only_clean = True

    # 1. The current commit.
    _status, output = run("ls-files")
    tracked = {line for line in output.splitlines() if line.strip()}
    published_clean &= report(f"working tree: {len(tracked)} file(s) tracked", tracked)

    # 2. Every commit reachable from this branch. A file deleted three commits
    #    ago is still in the history, and still in everyone's clone.
    _status, output = run("log", "--pretty=format:", "--name-only", "HEAD")
    history = {line for line in output.splitlines() if line.strip()}
    _status, count = run("rev-list", "--count", "HEAD")
    published_clean &= report(
        f"this branch's history: {count.strip()} commit(s)", history
    )

    # 3. Optionally everything git can still reach, including other branches
    #    and commits only the reflog remembers. Reported separately, because a
    #    branch that is never pushed is a different problem from a published
    #    one -- it is on this machine either way, and so is the directory the
    #    file came from.
    if args.all:
        _status, output = run("rev-list", "--objects", "--all", "--reflog")
        everywhere = {
            line.split(" ", 1)[1]
            for line in output.splitlines()
            if " " in line
        }
        local_only_clean &= report("every branch and the reflog", everywhere)

    # 4. What the remote actually has, which is the only thing that is really
    #    published. Asked of GitHub rather than of the local copy of it.
    if args.remote:
        _status, output = run("remote", "get-url", "origin")
        url = output.strip()
        slug = url.removesuffix(".git").split("github.com")[-1].lstrip(":/") if url else ""
        if not slug:
            print("  --      no origin remote to check")
        else:
            result = subprocess.run(
                ["gh", "api", f"repos/{slug}/git/trees/HEAD?recursive=1",
                 "--jq", ".tree[] | select(.type==\"blob\") | .path"],
                capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                print(f"  --      could not reach GitHub ({result.stderr.strip()[:80]})")
            else:
                remote = {line for line in result.stdout.splitlines() if line.strip()}
                published_clean &= report(
                    f"{slug} on GitHub: {len(remote)} file(s)", remote
                )

    audio = sorted(
        path for path in tracked
        if path.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
    )
    print(f"\n  Audio files committed: {len(audio)}")
    for path in audio:
        print(f"    {path}")

    local_stores()

    print()
    if not published_clean:
        print("Something private is in what this branch publishes.")
        print("Removing it from the working tree is not enough -- the blob stays in")
        print("the history, and in every clone. See README.md, 'What stays on your")
        print("machine'.")
        return 1
    if not local_only_clean:
        _status, branches = run("branch", "--format=%(refname:short)")
        print("Nothing private is published.")
        print()
        print("Some private paths do exist in branches other than this one. Those")
        print("branches are not what gets pushed, so nothing has left this machine --")
        print("but the blobs are in .git, and would go along if one were ever pushed.")
        print()
        print("  branches here:", ", ".join(branches.split()) or "(none)")
        print()
        print("  To be rid of them, delete every branch that is not the one you")
        print("  publish, then let git collect what nothing reaches any more:")
        print("      git branch -D <each other branch>")
        print("      git reflog expire --expire=now --all")
        print("      git gc --prune=now --aggressive")
        print("  That is irreversible: those commits are gone afterwards.")
        return 1
    print("Nothing private is committed, and nothing private is published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
