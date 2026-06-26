"""
Tests that verify blocking ASR operations run in a thread pool
and do not block the async event loop.
"""
import asyncio
import io
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


FAKE_AUDIO = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100
FAKE_AUDIO_NP = np.zeros(16000, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_to_thread_tracker():
    """Returns (tracking coroutine, call list) pair."""
    real_to_thread = asyncio.to_thread
    calls = []

    async def tracking(fn, *args, **kwargs):
        calls.append(fn)
        return await real_to_thread(fn, *args, **kwargs)

    return tracking, calls


# ---------------------------------------------------------------------------
# Behavior: transcribe runs via asyncio.to_thread (not in the event loop)
# ---------------------------------------------------------------------------


def test_transcribe_runs_in_thread(client):
    """load_audio and transcribe must be dispatched via asyncio.to_thread."""
    c, model = client
    model.transcribe.return_value = io.StringIO("hello world")
    tracking, calls = _make_to_thread_tracker()

    with (
        patch("app.webservice.load_audio", return_value=FAKE_AUDIO_NP),
        patch("app.webservice.asyncio.to_thread", side_effect=tracking),
    ):
        response = c.post(
            "/asr",
            files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )

    assert response.status_code == 200
    assert model.transcribe in calls, f"transcribe not dispatched via to_thread; calls: {calls}"


def test_load_audio_runs_in_thread_for_asr(client):
    """load_audio must be dispatched via asyncio.to_thread in the /asr endpoint."""
    c, model = client
    model.transcribe.return_value = io.StringIO("hello world")
    tracking, calls = _make_to_thread_tracker()

    with (
        patch("app.webservice.load_audio", return_value=FAKE_AUDIO_NP) as mock_load,
        patch("app.webservice.asyncio.to_thread", side_effect=tracking),
    ):
        response = c.post(
            "/asr",
            files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )

    assert response.status_code == 200
    assert mock_load in calls, f"load_audio not dispatched via to_thread; calls: {calls}"


# ---------------------------------------------------------------------------
# Behavior: detect-language runs via asyncio.to_thread
# ---------------------------------------------------------------------------


def test_detect_language_runs_in_thread(client):
    """load_audio and language_detection must be dispatched via asyncio.to_thread."""
    c, model = client
    model.language_detection.return_value = ("en", 0.95)
    tracking, calls = _make_to_thread_tracker()

    with (
        patch("app.webservice.load_audio", return_value=FAKE_AUDIO_NP) as mock_load,
        patch("app.webservice.asyncio.to_thread", side_effect=tracking),
    ):
        response = c.post(
            "/detect-language",
            files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )

    assert response.status_code == 200
    assert mock_load in calls, f"load_audio not dispatched via to_thread; calls: {calls}"
    assert model.language_detection in calls, f"language_detection not dispatched via to_thread; calls: {calls}"


# ---------------------------------------------------------------------------
# Behavior: /asr response has correct headers and body
# ---------------------------------------------------------------------------


def test_asr_response_contains_transcription(client):
    """The /asr endpoint must return the transcription text with correct headers."""
    c, model = client
    model.transcribe.return_value = io.StringIO("hello world")

    with patch("app.webservice.load_audio", return_value=FAKE_AUDIO_NP):
        response = c.post(
            "/asr",
            files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )

    assert response.status_code == 200
    assert "hello world" in response.text
    assert response.headers["content-type"].startswith("text/plain")
    assert "audio.wav" in response.headers.get("content-disposition", "")


# ---------------------------------------------------------------------------
# Behavior: /detect-language returns language and confidence
# ---------------------------------------------------------------------------


def test_decode_concurrency_is_bounded(client):
    """Concurrent /asr uploads must not decode more clips at once than DECODE_CONCURRENCY,
    even though transcription is serialized by a separate (smaller) semaphore. This is the
    host-RAM guard: without it, N concurrent decodes hold N full audio arrays in RAM."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    c, model = client
    model.transcribe.return_value = io.StringIO("hello world")

    import app.webservice as ws
    limit = ws._decode_semaphore._value  # initial = CONFIG.DECODE_CONCURRENCY

    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def blocking_load_audio(*_args, **_kwargs):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.1)  # hold the decode slot so overlap is observable
        with lock:
            state["now"] -= 1
        return FAKE_AUDIO_NP

    def fire():
        return c.post(
            "/asr",
            files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )

    with patch("app.webservice.load_audio", side_effect=blocking_load_audio):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = [f.result() for f in [pool.submit(fire) for _ in range(8)]]

    assert all(r.status_code == 200 for r in results)
    assert state["max"] <= limit, f"decode concurrency {state['max']} exceeded cap {limit}"
    if limit > 1:
        assert state["max"] > 1, "decode never ran in parallel; cap is not just serializing"


def test_detect_language_returns_language_and_confidence(client):
    """/detect-language must return detected_language, language_code, and confidence."""
    c, model = client
    model.language_detection.return_value = ("es", 0.97)

    with patch("app.webservice.load_audio", return_value=FAKE_AUDIO_NP):
        response = c.post(
            "/detect-language",
            files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["language_code"] == "es"
    assert body["detected_language"] == "spanish"
    assert body["confidence"] == pytest.approx(0.97)
