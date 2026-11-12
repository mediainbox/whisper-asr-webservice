"""Streaming separation: run_separation_gpu writes raw f32le chunk by chunk and returns the peak."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app import voice_separation as vs

SR = 44100


def _cfg(target="vocals"):
    return vs.VocalSeparationConfig(
        name="fake", sample_rate=SR, n_fft=6144, hop=1024, window="hann",
        dim_c=4, dim_f=2560, dim_t=8, target=target,
    )


def _chunks():
    rng = np.random.default_rng(0)
    return [rng.standard_normal((2, n)).astype(np.float32) * 0.5 for n in (5, 3, 4)]


def _fake_iter(chunks):
    def gen(*_args, **_kwargs):
        pos = 0
        for c in chunks:
            yield pos, c.copy()
            pos += c.shape[1]
    return gen


def _run(tmp_path, audio, chunks, target="vocals"):
    out = tmp_path / "vocals.f32"
    with patch.object(vs, "iter_separated_chunks", side_effect=_fake_iter(chunks)):
        peak = vs.run_separation_gpu(audio, _cfg(target), MagicMock(), "cpu", out)
    written = np.fromfile(out, dtype=np.float32).reshape(-1, 2).T
    return peak, written


def test_writes_chunks_as_interleaved_f32le(tmp_path):
    chunks = _chunks()
    audio = np.zeros((2, 12), dtype=np.float32)
    _, written = _run(tmp_path, audio, chunks)
    np.testing.assert_array_equal(written, np.concatenate(chunks, axis=-1))


def test_instrumental_target_writes_mix_minus_chunk(tmp_path):
    chunks = _chunks()
    audio = np.random.default_rng(1).standard_normal((2, 12)).astype(np.float32)
    _, written = _run(tmp_path, audio, chunks, target="instrumental")
    np.testing.assert_array_equal(written, audio - np.concatenate(chunks, axis=-1))


def test_returns_peak_including_negative(tmp_path):
    chunks = [np.full((2, 2), 0.3, np.float32), np.array([[0.1, -2.5], [1.2, 0.0]], np.float32)]
    peak, _ = _run(tmp_path, np.zeros((2, 4), np.float32), chunks)
    assert isinstance(peak, float)
    assert peak == pytest.approx(2.5)


def test_works_with_readonly_memmap(tmp_path):
    chunks = _chunks()
    mix = np.random.default_rng(2).standard_normal((2, 12)).astype(np.float32)
    raw = tmp_path / "mix.f32"
    np.ascontiguousarray(mix.T).tofile(raw)
    audio = np.memmap(raw, dtype=np.float32, mode="r").reshape(-1, 2).T
    _, written = _run(tmp_path, audio, chunks, target="instrumental")
    np.testing.assert_array_equal(written, mix - np.concatenate(chunks, axis=-1))


def test_load_audio_for_separation_rejects_empty_decode(tmp_path):
    in_path = tmp_path / "in.wav"
    in_path.write_bytes(b"x")
    fake_ffmpeg = MagicMock()

    def output(path, **_kwargs):
        Path(path).write_bytes(b"")  # ffmpeg "succeeded" but decoded nothing
        return MagicMock()

    fake_ffmpeg.input.return_value.output.side_effect = output
    with patch.object(vs, "ffmpeg", fake_ffmpeg), patch.object(
        vs, "get_model_and_config", return_value=(MagicMock(), _cfg())
    ):
        with pytest.raises(RuntimeError, match="no audio"):
            vs.load_audio_for_separation(in_path, device="cpu")


def test_load_audio_for_separation_returns_readonly_memmap(tmp_path):
    in_path = tmp_path / "in.wav"
    in_path.write_bytes(b"x")
    interleaved = np.arange(8, dtype=np.float32)  # L0 R0 L1 R1 ...
    fake_ffmpeg = MagicMock()

    def output(path, **_kwargs):
        interleaved.tofile(path)
        return MagicMock()

    fake_ffmpeg.input.return_value.output.side_effect = output
    with patch.object(vs, "ffmpeg", fake_ffmpeg), patch.object(
        vs, "get_model_and_config", return_value=(MagicMock(), _cfg())
    ):
        audio, cfg, _, device = vs.load_audio_for_separation(in_path, device="cpu")

    assert isinstance(audio, np.memmap) and not audio.flags.writeable
    np.testing.assert_array_equal(audio, interleaved.reshape(-1, 2).T)
    assert Path(audio.filename) == tmp_path / "in.wav.f32"
    assert cfg.sample_rate == SR and device == "cpu"
