import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.asr_models import blocked_features as bf


class FakeExtractor:
    """Pure-numpy copy of faster_whisper.feature_extractor.FeatureExtractor (faster-whisper 1.1.1),
    trimmed to the 1-D real-input path that transcribe.py uses."""

    def __init__(self, feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.time_per_frame = hop_length / sampling_rate
        self.sampling_rate = sampling_rate
        rng = np.random.default_rng(1)
        self.mel_filters = rng.uniform(0, 0.01, (feature_size, n_fft // 2 + 1)).astype("float32")
        self.calls = 0

    @staticmethod
    def stft(x, n_fft, hop_length, window, center=True, return_complex=True):
        if center:
            x = np.pad(x, n_fft // 2, mode="reflect")
        n_frames = 1 + (len(x) - n_fft) // hop_length
        frames = np.lib.stride_tricks.as_strided(x, (n_frames, n_fft), (hop_length * x.strides[0], x.strides[0]))
        return np.fft.rfft(frames * window, n=n_fft, axis=-1).T

    def __call__(self, waveform, padding=160, chunk_length=None):
        self.calls += 1
        if chunk_length is not None:
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length
        waveform = waveform.astype(np.float32)
        if padding:
            waveform = np.pad(waveform, (0, padding))
        window = np.hanning(self.n_fft + 1)[:-1].astype("float32")
        stft = self.stft(waveform, self.n_fft, self.hop_length, window=window).astype("complex64")
        magnitudes = np.abs(stft[..., :-1]) ** 2
        log_spec = np.log10(np.clip(self.mel_filters @ magnitudes, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0


def audio(n):
    return np.random.default_rng(n).standard_normal(n).astype(np.float32)


BLOCK = 10  # frames -> 1600 samples per block


@pytest.mark.parametrize("n", [5_003, 3 * BLOCK * 160, BLOCK * 160 + 1, 20_037])
@pytest.mark.parametrize("padding", [160, 0])
def test_long_audio_matches_reference(monkeypatch, n, padding):
    monkeypatch.setattr(bf, "BLOCK_FRAMES", BLOCK)
    inner = FakeExtractor()
    wrapped = bf.BlockedFeatureExtractor(inner)
    x = audio(n)

    got = wrapped(x, padding=padding)

    assert inner.calls == 0
    want = inner(x, padding=padding)
    assert got.shape == want.shape
    assert got.dtype == want.dtype == np.float32
    # BLAS may round the mel matmul differently by matrix width (1 ulp on OpenBLAS).
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_long_audio_chunk_length_updates_inner(monkeypatch):
    monkeypatch.setattr(bf, "BLOCK_FRAMES", BLOCK)
    inner = FakeExtractor()
    wrapped = bf.BlockedFeatureExtractor(inner)

    wrapped(audio(5_003), chunk_length=10)

    assert inner.n_samples == 160_000 and inner.nb_max_frames == 1_000
    assert wrapped.n_samples == 160_000


def test_short_audio_goes_straight_to_inner(monkeypatch):
    monkeypatch.setattr(bf, "BLOCK_FRAMES", BLOCK)
    monkeypatch.setattr(bf.BlockedFeatureExtractor, "_blocked", MagicMock(side_effect=AssertionError))
    inner = FakeExtractor()
    x = audio(BLOCK * 160)

    got = bf.BlockedFeatureExtractor(inner)(x, chunk_length=30)

    assert inner.calls == 1
    assert np.array_equal(got, FakeExtractor()(x))


def test_attributes_are_delegated():
    inner = FakeExtractor(feature_size=128)
    wrapped = bf.BlockedFeatureExtractor(inner)
    for name in ["sampling_rate", "hop_length", "n_fft", "nb_max_frames", "time_per_frame", "n_samples", "chunk_length"]:
        assert getattr(wrapped, name) == getattr(inner, name)
    assert wrapped.mel_filters is inner.mel_filters


def test_self_check_true_for_faithful_extractor():
    assert bf.blocked_matches_inner(FakeExtractor()) is True


@pytest.mark.parametrize("offset, expected", [(1e-7, True), (1e-3, False)])
def test_self_check_tolerates_rounding_but_not_real_differences(offset, expected):
    class Off(FakeExtractor):
        def __call__(self, *a, **k):
            return super().__call__(*a, **k) + np.float32(offset)

    assert bf.blocked_matches_inner(Off()) is expected


def test_self_check_false_when_shape_differs():
    class Short(FakeExtractor):
        def __call__(self, *a, **k):
            return super().__call__(*a, **k)[:, :-1]

    assert bf.blocked_matches_inner(Short()) is False


def test_self_check_false_when_extractor_raises():
    class Broken(FakeExtractor):
        def __call__(self, *a, **k):
            raise RuntimeError("boom")

    assert bf.blocked_matches_inner(Broken()) is False


def test_wrap_feature_extractor_wraps_when_check_passes():
    model = MagicMock()
    inner = FakeExtractor()
    model.feature_extractor = inner

    bf.wrap_feature_extractor(model)

    assert isinstance(model.feature_extractor, bf.BlockedFeatureExtractor)
    assert model.feature_extractor.inner is inner


def test_wrap_feature_extractor_keeps_original_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(bf, "blocked_matches_inner", lambda inner: False)
    model = MagicMock()
    inner = FakeExtractor()
    model.feature_extractor = inner

    with caplog.at_level(logging.WARNING, logger=bf.__name__):
        bf.wrap_feature_extractor(model)

    assert model.feature_extractor is inner
    assert any(r.levelno == logging.WARNING for r in caplog.records)
