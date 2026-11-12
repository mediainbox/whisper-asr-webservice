"""Unit tests for app.utils.load_audio numeric conversion."""
import io
from unittest.mock import MagicMock, patch

import numpy as np

from app import utils
from app.utils import load_audio, load_audio_path


def test_load_audio_converts_int16_to_normalized_float32():
    """encode=False path: int16 PCM bytes -> 1-D float32 array normalized to [-1, 1)."""
    samples = np.array([0, 32767, -32768, 16384], dtype=np.int16)
    out = load_audio(io.BytesIO(samples.tobytes()), encode=False)

    assert out.dtype == np.float32
    assert out.ndim == 1
    np.testing.assert_allclose(
        out, samples.astype(np.float32) / 32768.0, rtol=0, atol=0
    )


def test_load_audio_path_decodes_from_path_with_gain_and_shared_conversion():
    """load_audio_path hands ffmpeg the path (no read into RAM), applies the gain filter,
    and reuses load_audio's int16 -> float32 conversion."""
    samples = np.array([0, 32767, -32768, 16384], dtype=np.int16)
    fake_ffmpeg = MagicMock()
    stream = fake_ffmpeg.input.return_value
    stream.filter.return_value.output.return_value.run.return_value = (samples.tobytes(), b"")

    with patch.object(utils, "ffmpeg", fake_ffmpeg), patch.object(
        utils, "_pcm16_to_float32", wraps=utils._pcm16_to_float32
    ) as conv:
        out = load_audio_path(
            "/x/vocals.f32", sr=16000, input_kwargs={"format": "f32le", "ac": 2, "ar": 44100}, gain=0.5
        )

    fake_ffmpeg.input.assert_called_once_with("/x/vocals.f32", threads=0, format="f32le", ac=2, ar=44100)
    stream.filter.assert_called_once_with("volume", 0.5)
    stream.filter.return_value.output.assert_called_once_with(
        "-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000
    )
    conv.assert_called_once()
    assert out.dtype == np.float32 and out.ndim == 1
    np.testing.assert_allclose(out, samples.astype(np.float32) / 32768.0, rtol=0, atol=0)


def test_load_audio_path_without_gain_skips_volume_filter():
    fake_ffmpeg = MagicMock()
    stream = fake_ffmpeg.input.return_value
    stream.output.return_value.run.return_value = (np.zeros(2, np.int16).tobytes(), b"")

    with patch.object(utils, "ffmpeg", fake_ffmpeg):
        out = load_audio_path("/x/a.wav")

    stream.filter.assert_not_called()
    assert out.shape == (2,)
