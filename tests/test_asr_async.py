"""
Tests that verify blocking ASR operations run in a thread pool
and do not block the async event loop.
"""
import asyncio
import io
import time
from pathlib import Path
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


def test_requests_active_not_leaked_on_exception(client):
    """A crash mid-request (e.g. the CUDA OOMs seen in prod) must not leak
    requests.active — the counter should return to 0 once the request unwinds."""
    c, model = client
    model.transcribe.side_effect = RuntimeError("boom")

    with patch("app.webservice.load_audio", return_value=FAKE_AUDIO_NP):
        with pytest.raises(RuntimeError):
            c.post(
                "/asr",
                files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
            )

    assert c.get("/stats").json()["requests"]["active"] == 0


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

    # Widen the transcribe gate (1 in tests) so any 503 could only come from decode.
    with patch("app.webservice.load_audio", side_effect=blocking_load_audio), patch.object(
        ws, "_transcribe_semaphore", asyncio.Semaphore(8)
    ):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = [f.result() for f in [pool.submit(fire) for _ in range(8)]]

    # Decodes are short, so requests beyond capacity wait for a slot (DECODE_WAIT_SECONDS)
    # instead of bouncing with 503.
    assert [r.status_code for r in results] == [200] * 8
    assert state["max"] <= limit, f"decode concurrency {state['max']} exceeded cap {limit}"
    if limit > 1:
        assert state["max"] > 1, "decode never ran in parallel; cap is not just serializing"


def test_decode_gate_rejects_after_wait(client):
    """When no decode slot frees up within DECODE_WAIT_SECONDS, the request gets 503 with
    Retry-After, and the active-request counter is still released."""
    from concurrent.futures import ThreadPoolExecutor

    import app.webservice as ws

    c, model = client
    model.transcribe.return_value = io.StringIO("hello world")
    n = ws._decode_semaphore._value + 2

    def slow_load_audio(*_args, **_kwargs):
        time.sleep(0.5)
        return FAKE_AUDIO_NP

    def fire():
        return c.post(
            "/asr",
            files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )

    with patch("app.webservice.load_audio", side_effect=slow_load_audio), patch.object(
        ws.CONFIG, "DECODE_WAIT_SECONDS", 0.1
    ):
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = [f.result() for f in [pool.submit(fire) for _ in range(n)]]

    rejected = [r for r in results if r.status_code == 503]
    assert rejected, "expected 503 once the wait expired"
    assert all(r.headers.get("retry-after") == "5" for r in rejected)
    assert all(r.status_code in (200, 503) for r in results)
    assert c.get("/stats").json()["requests"]["active"] == 0
    assert ws._decode_semaphore._value == ws.CONFIG.DECODE_CONCURRENCY


def test_stats_exposes_decode_os_and_gpu(client):
    """/stats must expose the decode semaphore block, OS metrics, and a gpu list."""
    from app.config import CONFIG

    c, _ = client
    response = c.get("/stats")

    assert response.status_code == 200
    body = response.json()

    # decode block mirrors transcribe/vocals and reflects the dedicated semaphore
    assert body["decode"]["concurrency"] == CONFIG.DECODE_CONCURRENCY
    assert body["decode"]["slots_used"] == 0  # idle: nothing decoding
    assert body["decode"]["slots_free"] == CONFIG.DECODE_CONCURRENCY

    # OS metrics present and best-effort (load_avg works cross-platform)
    assert isinstance(body["os"], dict)
    assert "load_avg" in body["os"]

    # gpu list present (empty when cuda unavailable, as in tests)
    assert isinstance(body["gpu"], list)


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


def _fire_vocals_burst(c, ws, n, separation_s, vocals_wait_s):
    """n concurrent /asr?separate_vocals=true with the transcribe gate widened, so a 503
    can only come from the vocals gate."""
    from concurrent.futures import ThreadPoolExecutor

    def fake_separation(_audio, _cfg, _model, _device, out_path, **_kwargs):
        time.sleep(separation_s)
        Path(out_path).write_bytes(FAKE_AUDIO)

    def fire():
        return c.post(
            "/asr?separate_vocals=true",
            files={"audio_file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )

    with patch("app.webservice.load_audio_for_separation", return_value=(FAKE_AUDIO_NP, None, None, "cpu")), patch(
        "app.webservice.run_separation_gpu", side_effect=fake_separation
    ), patch("app.webservice.load_audio", return_value=FAKE_AUDIO_NP), patch.object(
        ws, "_transcribe_semaphore", asyncio.Semaphore(n)
    ), patch.object(ws.CONFIG, "VOCALS_WAIT_SECONDS", vocals_wait_s):
        with ThreadPoolExecutor(max_workers=n) as pool:
            return [f.result() for f in [pool.submit(fire) for _ in range(n)]]


def test_vocals_gate_waits_for_slot(client):
    """Requests beyond VOCALS_CONCURRENCY wait for a separation slot instead of getting 503."""
    import app.webservice as ws

    c, model = client
    model.transcribe.side_effect = lambda *a, **k: io.StringIO("hello world")
    n = ws._vocals_semaphore._value + 2

    results = _fire_vocals_burst(c, ws, n, separation_s=0.1, vocals_wait_s=30)

    assert [r.status_code for r in results] == [200] * n


def test_vocals_gate_rejects_after_wait(client):
    """When no separation slot frees up within VOCALS_WAIT_SECONDS: 503 with Retry-After,
    and no slot or active-request count is leaked."""
    import app.webservice as ws

    c, model = client
    model.transcribe.side_effect = lambda *a, **k: io.StringIO("hello world")
    n = ws._vocals_semaphore._value + 2

    results = _fire_vocals_burst(c, ws, n, separation_s=0.5, vocals_wait_s=0.1)

    rejected = [r for r in results if r.status_code == 503]
    assert rejected, "expected 503 once the wait expired"
    assert all(r.headers.get("retry-after") == "5" for r in rejected)
    assert all(r.status_code in (200, 503) for r in results)
    assert c.get("/stats").json()["requests"]["active"] == 0
    assert ws._vocals_semaphore._value == ws.CONFIG.VOCALS_CONCURRENCY
