"""Unit tests for app.utils.load_audio numeric conversion."""
import io

import numpy as np

from app.utils import load_audio


def test_load_audio_converts_int16_to_normalized_float32():
    """encode=False path: int16 PCM bytes -> 1-D float32 array normalized to [-1, 1)."""
    samples = np.array([0, 32767, -32768, 16384], dtype=np.int16)
    out = load_audio(io.BytesIO(samples.tobytes()), encode=False)

    assert out.dtype == np.float32
    assert out.ndim == 1
    np.testing.assert_allclose(
        out, samples.astype(np.float32) / 32768.0, rtol=0, atol=0
    )
