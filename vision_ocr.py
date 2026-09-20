"""A screenshot, turned into the text a person actually wanted read to them.

Plain OCR is the wrong tool here and it is worth being clear why. A phone
screenshot is mostly not prose: there is a clock and a battery icon along the
top, a navigation bar along the bottom, a row of tabs, a like button, a cookie
banner, the name of the app. Read it all out in order and the first thing you
hear is "9:41". What is wanted is the article, the message, the post -- and
knowing which part of a picture is *that* is a judgement, not a transcription.

So the first pass is a small multimodal model running locally through LM Studio,
which is given the picture and told what to keep. It is not being asked to be
clever; it is being asked to leave things out, which is the one thing a
character recogniser cannot do.

Two fallbacks behind it, in order:

*macOS's own Vision framework*, if LM Studio is not running. It is built into
the system, needs no model loaded and no memory, and is very good at getting the
characters right -- it simply has no opinion about which of them matter. Its
output is marked as unfiltered so the caller can say so.

*Nothing.* If neither is available the caller is told which one to start, rather
than being handed an empty read.

Whatever comes back is offered to the person for a glance and an edit before it
is spoken, because no amount of prompting makes "which part of this did you
mean" reliable, and a two-second look is cheaper than listening to the wrong
thing.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import requests

logger = logging.getLogger("breeze.vision")

# LM Studio's local server, in its default place. Anything that speaks the
# OpenAI chat-completions shape works; this is only where we look first.
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"

# How long to wait for the model. A vision pass on a phone screenshot is a few
# seconds on this hardware; a minute means something is wrong, not slow.
TIMEOUT_SECONDS = 90

# Bigger than any screenshot a phone produces. A limit, not a target: the point
# is to refuse a video frame or a scan, not to compress anything.
MAX_IMAGE_BYTES = 12 * 1024 * 1024

PROMPT = (
    "This is a screenshot from a phone. Transcribe only the readable body "
    "content a person would want read aloud to them: the article, message, "
    "post or document text.\n\n"
    "Leave out everything that is interface rather than content: the status "
    "bar, clock, battery and signal indicators, navigation and tab bars, "
    "buttons, menus, search boxes, timestamps that belong to the app rather "
    "than the text, share and like counts, adverts, cookie notices, and "
    "'related articles' lists.\n\n"
    "Keep the wording exactly as written and keep paragraph breaks. Do not "
    "summarise, translate, explain or add headings of your own. Output the "
    "text and nothing else. If there is no body text in the image, output "
    "nothing at all."
)


def _enabled() -> bool:
    return (os.getenv("BREEZE_VISION_ENABLED", "1") or "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _base_url() -> str:
    return (os.getenv("BREEZE_VISION_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _configured_model() -> str | None:
    model = (os.getenv("BREEZE_VISION_MODEL") or "").strip()
    return model or None


def _loaded_models(timeout: float = 2.0) -> list[str]:
    """What the local server currently has loaded, or an empty list if it is not up."""
    try:
        response = requests.get(f"{_base_url()}/models", timeout=timeout)
        response.raise_for_status()
        return [
            str(item.get("id"))
            for item in (response.json().get("data") or [])
            if item.get("id")
        ]
    except Exception:  # noqa: BLE001 - "not running" is the expected failure
        return []


def _pick_model() -> str | None:
    """The model to send to: the configured one, else whatever looks like a VLM."""
    configured = _configured_model()
    if configured:
        return configured
    loaded = _loaded_models()
    for name in loaded:
        lowered = name.lower()
        if any(hint in lowered for hint in ("vl", "vision", "llava", "pixtral")):
            return name
    return loaded[0] if loaded else None


def status() -> dict[str, Any]:
    """What the settings page shows: whether a screenshot can be read, and why not."""
    if not _enabled():
        return {
            "available": False,
            "backend": None,
            "reason": "Screenshot reading is switched off (BREEZE_VISION_ENABLED=0)",
        }
    model = _pick_model()
    if model:
        return {
            "available": True,
            "backend": "lmstudio",
            "model": model,
            "base_url": _base_url(),
            "reason": None,
        }
    if _macos_vision_available():
        return {
            "available": True,
            "backend": "macos-vision",
            "model": None,
            "base_url": None,
            "reason": (
                "LM Studio is not answering; falling back to macOS text "
                "recognition, which reads everything including the clock"
            ),
        }
    return {
        "available": False,
        "backend": None,
        "reason": (
            f"No local vision model: start LM Studio's server at {_base_url()} "
            "and load a vision model (a Qwen-VL works well)"
        ),
    }


def extract(image: bytes, media_type: str = "image/png") -> dict[str, Any]:
    """The readable text in a screenshot, and which backend found it.

    Returns ``{text, backend, filtered, model}``. ``filtered`` says whether
    interface chrome was deliberately left out -- false means the caller should
    expect a clock in the first line and show the text for editing.
    """
    if not image:
        raise ValueError("No image data")
    if len(image) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"That image is {len(image) // (1024 * 1024)} MB; "
            f"the limit is {MAX_IMAGE_BYTES // (1024 * 1024)} MB"
        )

    if _enabled():
        model = _pick_model()
        if model:
            try:
                return {
                    "text": _ask_model(model, image, media_type),
                    "backend": "lmstudio",
                    "model": model,
                    "filtered": True,
                }
            except Exception:  # noqa: BLE001 - fall through to the system OCR
                logger.exception("The local vision model failed; trying macOS Vision")

    text = _macos_vision(image)
    if text is not None:
        return {
            "text": text,
            "backend": "macos-vision",
            "model": None,
            "filtered": False,
        }
    raise RuntimeError(status()["reason"] or "No way to read this screenshot")


def _ask_model(model: str, image: bytes, media_type: str) -> str:
    data_uri = f"data:{media_type};base64,{base64.b64encode(image).decode()}"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }
    response = requests.post(
        f"{_base_url()}/chat/completions", json=payload, timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    choices = response.json().get("choices") or []
    if not choices:
        raise RuntimeError("The vision model returned nothing")
    return (choices[0].get("message") or {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# The system's own text recognition
# ---------------------------------------------------------------------------
def _macos_vision_available() -> bool:
    try:
        import Vision  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _macos_vision(image: bytes) -> str | None:
    """Every line macOS can read in the image, top to bottom. None if unavailable.

    No judgement about what matters -- that is the point of the model above --
    but it is accurate, instant, and already on the machine.
    """
    try:
        import Quartz
        import Vision
        from Foundation import NSData
    except Exception:  # noqa: BLE001 - pyobjc is optional
        return None
    try:
        data = NSData.dataWithBytes_length_(image, len(image))
        source = Quartz.CGImageSourceCreateWithData(data, None)
        if source is None:
            return None
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
        if cg_image is None:
            return None

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        handler.performRequests_error_([request], None)

        lines: list[tuple[int, float, str]] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            box = observation.boundingBox()
            # Vision reports each run of text with its own box, in no
            # particular order, with the origin at the bottom left -- so a
            # larger y is further up the page. Sorting by y alone puts two
            # things on the same line in whichever order they were found,
            # which is how "9:41" ends up after the battery icon. Rounding y
            # into bands first makes a line a line, and then x reads it.
            row = int(round((1.0 - box.origin.y) * 80))
            lines.append((row, box.origin.x, candidates[0].string()))
        lines.sort(key=lambda item: (item[0], item[1]))
        return "\n".join(text for _, _, text in lines).strip()
    except Exception:  # noqa: BLE001 - a failed OCR is reported, not raised
        logger.exception("macOS text recognition failed")
        return None
