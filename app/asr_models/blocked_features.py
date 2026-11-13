"""Log-mel features computed in blocks of frames, matching faster-whisper's FeatureExtractor.

WhisperModel.transcribe computes the features of the whole (post-VAD) clip in one call, which
allocates ~64 bytes of transient RAM per audio sample (framed windows plus FFT temporaries):
~6 GiB for a 1 h 51 min clip. Every STFT frame is independent and the only global coupling is the
`max - 8` clamp, so computing blocks of frames and clamping at the end gives the same features up
to float32 rounding: BLAS may round the `mel_filters @ magnitudes` product differently depending on
the matrix width (OpenBLAS: <= 1 ulp, ~6e-8, in ~0.02 % of the values; macOS/Accelerate: identical).
The model consumes the features in float16, whose resolution is ~4 orders of magnitude coarser.

Only depends on numpy so it can be tested without faster_whisper installed.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 120 s of audio at 16 kHz / hop 160. Peak RAM is ~ the output array plus one block's temporaries.
# Linux/OpenBLAS, 4 CPUs, 25 min of audio (time / peak): all-at-once 0.83 s / 1.43 GiB; blocks of
# 15 s 4.70 s / 0.09 GiB, 60 s 2.48 s / 0.15 GiB, 120 s 1.72 s / 0.23 GiB, 300 s 1.36 s / 0.46 GiB,
# 600 s 1.24 s / 0.85 GiB. 120 s costs ~2x the original's time in a stage that lasts a few seconds,
# with ~0.2 GiB of temporaries per block instead of ~64 bytes per sample of the whole clip.
BLOCK_FRAMES = 12000

# The self-check accepts float32 rounding differences from BLAS; a structural change in the library
# (window, padding, frame count) produces differences orders of magnitude larger.
TOLERANCE = 1e-6


class BlockedFeatureExtractor:
    """Wraps a faster_whisper FeatureExtractor; everything except __call__ is delegated to it."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, waveform: np.ndarray, padding=160, chunk_length=None):
        # Clips up to one block (the 30 s segments that make up most traffic) take the original path.
        if waveform.ndim != 1 or len(waveform) <= BLOCK_FRAMES * self.inner.hop_length:
            return self.inner(waveform, padding, chunk_length)
        if chunk_length is not None:
            self.inner.n_samples = chunk_length * self.inner.sampling_rate
            self.inner.nb_max_frames = self.inner.n_samples // self.inner.hop_length
        return self._blocked(waveform, padding, BLOCK_FRAMES)

    def _blocked(self, waveform, padding, block_frames):
        fe = self.inner
        x = np.asarray(waveform, dtype=np.float32)
        n = len(x)
        padded = n + (padding or 0)  # length after the original's np.pad(waveform, (0, padding))
        half = fe.n_fft // 2
        keep = (padded + 2 * half - fe.n_fft) // fe.hop_length  # STFT frames minus the dropped last one
        window = np.hanning(fe.n_fft + 1)[:-1].astype("float32")
        out = np.empty((fe.mel_filters.shape[0], keep), dtype=np.float32)
        for f0 in range(0, keep, block_frames):
            f1 = min(f0 + block_frames, keep)
            # Sample indices of these frames in the padded waveform, with the STFT's center reflect
            # applied only at the global edges; indices past the real audio are the zero padding.
            j = np.arange(f0 * fe.hop_length, (f1 - 1) * fe.hop_length + fe.n_fft) - half
            j = np.where(j < 0, -j, j)
            j = np.where(j >= padded, 2 * (padded - 1) - j, j)
            seg = np.where(j < n, x[np.minimum(j, n - 1)], np.float32(0)).astype(np.float32)
            stft = fe.stft(seg, fe.n_fft, fe.hop_length, window=window, center=False, return_complex=True)
            magnitudes = np.abs(stft.astype("complex64")) ** 2
            out[:, f0:f1] = np.log10(np.clip(fe.mel_filters @ magnitudes, a_min=1e-10, a_max=None))
        np.maximum(out, out.max() - 8.0, out=out)
        out += 4.0
        out /= 4.0
        return out


def blocked_matches_inner(inner) -> bool:
    """True if the blocked path matches `inner` (same shape and dtype, within TOLERANCE) on a short
    synthetic clip split into several blocks plus a partial tail."""
    try:
        x = np.random.default_rng(0).standard_normal(47_400).astype(np.float32)  # 297 frames = 8 x 37 + 1
        a, b = inner(x), BlockedFeatureExtractor(inner)._blocked(x, 160, 37)
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.allclose(a, b, rtol=0, atol=TOLERANCE))
    except Exception:
        return False


def wrap_feature_extractor(model):
    """Swap model.feature_extractor for the blocked one, unless the self-check says it would differ."""
    if blocked_matches_inner(model.feature_extractor):
        model.feature_extractor = BlockedFeatureExtractor(model.feature_extractor)
    else:
        logger.warning(
            "Blocked log-mel features do not match this faster-whisper FeatureExtractor; "
            "keeping the original (whole-clip features use ~64 bytes of RAM per audio sample)"
        )
