"""Call the local Breeze TTS 2 server from any codebase on this machine.

    python client_example.py
"""

from __future__ import annotations

import time

import numpy as np
import requests
import soundfile as sf

BASE_URL = "http://127.0.0.1:7860"
SAMPLE_RATE = 24000


def speak_to_wav(text: str, path: str, **fields) -> str:
    """Synthesize to a .wav file. Returns the path."""
    response = requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json={"text": text, **fields},
        timeout=1800,
    )
    response.raise_for_status()
    with open(path, "wb") as handle:  # response body is a complete WAV file
        handle.write(response.content)
    return path


def stream_pcm(text: str, **fields) -> np.ndarray:
    """Stream raw PCM for low-latency playback. Returns float32 mono audio.

    Note the server sends a bare PCM stream here -- signed 16-bit little-endian,
    24 kHz, mono, with no WAV header -- so it can be fed straight to an audio
    device as it arrives.
    """
    started = time.time()
    first_byte: float | None = None
    buffer = bytearray()

    with requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json={"text": text, "stream": True, **fields},
        stream=True,
        timeout=1800,
    ) as response:
        response.raise_for_status()
        for piece in response.iter_content(4096):
            if not piece:
                continue
            if first_byte is None:
                first_byte = time.time() - started
            buffer += piece

    pcm = np.frombuffer(bytes(buffer), dtype="<i2").astype(np.float32) / 32767.0
    print(
        f"  time to first audio: {first_byte:.2f}s | "
        f"{len(pcm) / SAMPLE_RATE:.2f}s audio in {time.time() - started:.2f}s"
    )
    return pcm


def clone_voice(text: str, ref_audio: str, ref_text: str, path: str) -> str:
    """Voice cloning takes the reference as a file upload (multipart)."""
    with open(ref_audio, "rb") as handle:
        response = requests.post(
            f"{BASE_URL}/v1/audio/speech",
            files={"ref_audio": handle},
            data={"text": text, "ref_text": ref_text},
            timeout=1800,
        )
    response.raise_for_status()
    with open(path, "wb") as out:
        out.write(response.content)
    return path


def llm_to_speech(sentence: str) -> np.ndarray:
    """The LLM-to-TTS loop: clean the text with Gemini, then speak it.

    ``prepare`` runs the sentence through Gemini Flash Lite on Vertex first,
    which fixes punctuation and adds supported vocal events. If Vertex is
    unreachable the server falls back to the raw text and reports the reason in
    the X-Breeze-Prep-Error header rather than failing the request.
    """
    return stream_pcm(
        sentence,
        instruction="A warm, natural conversational voice.",
        cfg_scale=4,
        prepare=True,
    )


if __name__ == "__main__":
    print("health:", requests.get(f"{BASE_URL}/health", timeout=10).json())

    print("\n1. Voice design ->")
    print("  ", speak_to_wav(
        "In a world where nothing is what it seems, one developer must ship before Friday.",
        "outputs/client_design.wav",
        instruction="A deep, booming movie-trailer narrator, dramatic and intense.",
        cfg_scale=4,
    ))

    print("\n2. Streaming ->")
    audio = stream_pcm(
        "Streaming keeps latency low, because playback starts before generation ends.",
        instruction="An upbeat narrator.",
        cfg_scale=4,
    )
    sf.write("outputs/client_stream.wav", audio, SAMPLE_RATE)

    print("\n3. LLM-prepared speech ->")
    audio = llm_to_speech(
        "ok so the build is green now i fixed the flaky test it was a race condition"
    )
    sf.write("outputs/client_prepared.wav", audio, SAMPLE_RATE)
    print("   wrote outputs/client_prepared.wav")
