import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import ffmpeg
import numpy as np
import torch
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class VocalSeparationConfig:
    name: str
    sample_rate: int
    n_fft: int
    hop: int
    window: str
    dim_c: int
    dim_f: int
    dim_t: int
    target: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VocalSeparationConfig":
        required = ["name", "sample_rate", "n_fft", "hop", "window", "dim_c", "dim_f", "dim_t", "target"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        target = str(d["target"]).lower()
        if target not in ("vocals", "instrumental"):
            raise ValueError("config.target must be 'vocals' or 'instrumental'")
        return VocalSeparationConfig(
            name=str(d["name"]),
            sample_rate=int(d["sample_rate"]),
            n_fft=int(d["n_fft"]),
            hop=int(d["hop"]),
            window=str(d["window"]),
            dim_c=int(d["dim_c"]),
            dim_f=int(d["dim_f"]),
            dim_t=int(d["dim_t"]),
            target=target,
        )

    def outputs_instrumental(self) -> bool:
        return self.target == "instrumental"


@lru_cache(maxsize=1)
def get_model_and_config(
    model_id: str = "UVR-MDX-NET-Inst_HQ_4",
    repo_id: str = "mediainbox/uvr-mdx-models",
    device: str = "cuda",
) -> tuple[torch.jit.ScriptModule, VocalSeparationConfig]:
    device = device.lower()
    model_path = hf_hub_download(repo_id, f"{model_id}.pt")
    cfg_path = hf_hub_download(repo_id, f"{model_id}.json")

    model = torch.jit.load(model_path, map_location=device).eval()
    with open(cfg_path, "r") as f:
        cfg = VocalSeparationConfig.from_dict(json.load(f))

    return model, cfg


def iter_separated_chunks(
    model: torch.jit.ScriptModule,
    input_audio: np.ndarray,
    device: str,
    n_fft: int,
    sample_rate: int,
    dim_f: int,
    hop: int,
    chunks: int = 30,
    use_tta: bool = False,
    precision: str = "fp16",
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Hacked from https://github.com/seanghay/vocal/blob/main/vocal/__init__.py.

    Yields (pos, chunk) in order: `chunk` is a float32 (2, m) slice of the model output with
    the overlap margins already trimmed, and `pos` is the sample offset of its first sample.
    Chunks are contiguous and non-overlapping, so their concatenation is the full output;
    only one chunk is alive at a time, keeping host RAM O(chunk) instead of O(duration).
    """
    device = device.lower()
    is_cuda = device.startswith("cuda")

    # Pre-compute constants
    audio_chunk_size = chunks * sample_rate
    dim_t = 2**8
    dim_c = 4
    chunk_size = hop * (dim_t - 1)
    n_bins = n_fft // 2 + 1

    # Move window to GPU once
    window = torch.hann_window(window_length=n_fft, periodic=True).to(device)

    # Pre-compute frequency padding
    out_c = dim_c
    _freq_pad = torch.zeros([1, out_c, n_bins - dim_f, dim_t], device=device)

    # Convert input to correct format
    if input_audio.ndim == 1:
        input_audio = np.asfortranarray([input_audio, input_audio])

    margin = sample_rate if sample_rate < audio_chunk_size else audio_chunk_size
    samples = input_audio.shape[-1]

    if chunks == 0 or samples < audio_chunk_size:
        audio_chunk_size = samples

    # Overlapping chunk bounds only; the mix itself stays where it is (possibly a memmap on
    # disk) and each chunk is copied to the device one at a time inside the loop, so neither
    # VRAM nor host RAM grows with clip duration.
    chunk_bounds = []
    for skip in range(0, samples, audio_chunk_size):
        s_margin = 0 if skip == 0 else margin
        end = min(skip + audio_chunk_size + margin, samples)
        start = skip - s_margin
        chunk_bounds.append((start, end))
        if end == samples:
            break

    margin_size = margin
    pos = 0

    amp_enabled = is_cuda and precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else (torch.bfloat16 if precision == "bf16" else None)

    for chunk_mix_position, (chunk_start, chunk_end) in enumerate(chunk_bounds):
        chunk_mix = torch.from_numpy(np.ascontiguousarray(input_audio[:, chunk_start:chunk_end])).to(device)
        n_sample = chunk_mix.shape[1]
        trim = n_fft // 2
        gen_size = chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        # Perform padding on GPU
        mix_p = torch.cat(
            [
                torch.zeros(2, trim, device=device),
                chunk_mix,
                torch.zeros(2, pad, device=device),
                torch.zeros(2, trim, device=device),
            ],
            dim=1,
        )

        # Process waves in batches
        mix_waves = []
        i = 0
        while i < n_sample + pad:
            waves = mix_p[:, i : i + chunk_size]
            mix_waves.append(waves)
            i += gen_size

        # Stack waves efficiently
        mix_waves = torch.stack(mix_waves)

        with torch.no_grad():
            # Process in a single forward pass
            x = mix_waves.reshape(-1, chunk_size)

            # Perform STFT
            x = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop,
                window=window,
                center=True,
                return_complex=True,
            )
            x = torch.view_as_real(x)
            x = x.permute(0, 3, 1, 2)

            # Reshape efficiently
            x = x.reshape(-1, 2, 2, n_bins, dim_t).reshape(-1, dim_c, n_bins, dim_t)
            x = x[:, :, :dim_f]

            # Model inference with memory optimization
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    spec_pred = (-model(-x) + model(x)) * 0.5 if use_tta else model(x)
            else:
                spec_pred = (-model(-x) + model(x)) * 0.5 if use_tta else model(x)

            # Post-processing on GPU
            spec_pred = spec_pred.float()
            x = torch.cat([spec_pred, _freq_pad.expand(spec_pred.shape[0], -1, -1, -1)], dim=2)
            c = 2
            x = x.reshape(-1, c, 2, n_bins, dim_t).reshape(-1, 2, n_bins, dim_t)
            x = x.permute(0, 2, 3, 1).contiguous()

            # Inverse STFT
            x = torch.view_as_complex(x)
            x = torch.istft(x, n_fft=n_fft, hop_length=hop, window=window, center=True)
            x = x.reshape(-1, c, chunk_size)

            # Move to CPU only at the end
            tar_waves = x.cpu()

            tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]

            start = 0 if chunk_mix_position == 0 else margin_size
            end = None if chunk_mix_position == len(chunk_bounds) - 1 else -margin_size
            if margin_size == 0:
                end = None

        chunk = tar_signal[:, start:end]
        yield pos, chunk
        pos += chunk.shape[1]


def separate_vocal(
    model: torch.jit.ScriptModule,
    input_audio: np.ndarray,
    device: str,
    n_fft: int,
    sample_rate: int,
    dim_f: int,
    hop: int,
    chunks: int = 30,
    use_tta: bool = False,
    precision: str = "fp16",
) -> np.ndarray:
    """In-memory variant of iter_separated_chunks: returns the whole (2, n) output."""
    return np.concatenate(
        [
            c
            for _, c in iter_separated_chunks(
                model, input_audio, device, n_fft, sample_rate, dim_f, hop, chunks, use_tta, precision
            )
        ],
        axis=-1,
    )


def separate_vocals_from_array(
    audio: np.ndarray,
    model_id: str = "UVR-MDX-NET-Inst_HQ_4",
    repo_id: str = "mediainbox/uvr-mdx-models",
    device: str | None = None,
    chunks: int = 30,
    use_tta: bool = False,
    precision: str = "fp16",
) -> np.ndarray:
    device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    model, cfg = get_model_and_config(
        model_id=model_id,
        repo_id=repo_id,
        device=device,
    )
    return separate_vocal(
        model=model,
        input_audio=audio,
        device=device,
        n_fft=cfg.n_fft,
        sample_rate=cfg.sample_rate,
        dim_f=cfg.dim_f,
        hop=cfg.hop,
        chunks=chunks,
        use_tta=use_tta,
        precision=precision,
    )


def _decode_audio_to_file(input_path: Path, raw_path: Path, sample_rate: int) -> None:
    """Decode any audio format to raw interleaved stereo f32le on disk via ffmpeg, so the
    full clip never has to sit in RAM (avoids soundfile/audioread limits too)."""
    try:
        (
            ffmpeg.input(str(input_path), threads=0)
            .output(str(raw_path), format="f32le", ac=2, ar=sample_rate)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to decode audio for separation: {e.stderr.decode()}") from e
    if raw_path.stat().st_size == 0:  # np.memmap cannot map an empty file
        raise RuntimeError(f"Failed to decode audio for separation: ffmpeg produced no audio from {input_path.name}")


def load_audio_for_separation(
    input_path: str | Path,
    model_id: str = "UVR-MDX-NET-Inst_HQ_4",
    repo_id: str = "mediainbox/uvr-mdx-models",
    device: str | None = None,
) -> tuple[np.ndarray, "VocalSeparationConfig", torch.jit.ScriptModule, str]:
    """
    CPU-only phase: decode audio and warm up the cached model handle.
    Call this before acquiring the GPU semaphore.
    Returns (audio, cfg, model, device) ready for run_separation_gpu. `audio` is a read-only
    (2, n) np.memmap over a raw file written next to `input_path` (`<name>.f32`), so it is
    cleaned up together with the caller's temp dir.
    """
    device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    model, cfg = get_model_and_config(model_id=model_id, repo_id=repo_id, device=device)
    input_path = Path(input_path)
    raw_path = input_path.with_name(input_path.name + ".f32")
    _decode_audio_to_file(input_path, raw_path, cfg.sample_rate)
    audio = np.memmap(raw_path, dtype=np.float32, mode="r").reshape(-1, 2).T
    return audio, cfg, model, device


def run_separation_gpu(
    audio: np.ndarray,
    cfg: "VocalSeparationConfig",
    model: torch.jit.ScriptModule,
    device: str,
    output_path: str | Path,
    chunks: int = 30,
    use_tta: bool = False,
    precision: str = "fp16",
) -> float:
    """
    GPU phase: run model inference and stream the vocals to `output_path` chunk by chunk.
    Call this while holding the GPU semaphore.

    `output_path` gets headerless raw PCM: interleaved stereo f32le at cfg.sample_rate,
    NOT peak-normalized (no WAV header: it would cap the file at 4 GB, and normalizing needs
    the global peak, which is only known at the end). Returns that peak (max |sample|); the
    reader applies the 1/peak gain when it is finite and > 1.0.
    """
    peak = 0.0
    with open(output_path, "wb") as f:
        for pos, chunk in iter_separated_chunks(
            model=model,
            input_audio=audio,
            device=device,
            n_fft=cfg.n_fft,
            chunks=chunks,
            sample_rate=cfg.sample_rate,
            dim_f=cfg.dim_f,
            hop=cfg.hop,
            use_tta=use_tta,
            precision=precision,
        ):
            if cfg.outputs_instrumental():
                # vocals = mix - instrumental, computed in the chunk's own buffer
                if not chunk.flags.writeable:
                    chunk = chunk.copy()
                np.subtract(audio[:, pos : pos + chunk.shape[1]], chunk, out=chunk)
            # np.max (not builtin max) so a NaN propagates like the old whole-array peak did
            peak = float(np.max([peak, chunk.max(), -chunk.min()]))
            np.ascontiguousarray(chunk.T).tofile(f)
    return peak


def separate_vocals_from_file(
    input_path: str | Path,
    output_path: str | Path,
    model_id: str = "UVR-MDX-NET-Inst_HQ_4",
    repo_id: str = "mediainbox/uvr-mdx-models",
    device: str | None = None,
    chunks: int = 30,
    use_tta: bool = False,
    precision: str = "fp16",
) -> None:
    """Separate vocals into `output_path` as a peak-normalized 16-bit PCM WAV."""
    input_path, output_path = Path(input_path), Path(output_path)
    raw_in = input_path.with_name(input_path.name + ".f32")  # written by load_audio_for_separation
    raw_out = output_path.with_name(output_path.name + ".f32")
    try:
        audio, cfg, model, device = load_audio_for_separation(
            input_path, model_id=model_id, repo_id=repo_id, device=device
        )
        peak = run_separation_gpu(audio, cfg, model, device, raw_out, chunks, use_tta, precision)
        stream = ffmpeg.input(str(raw_out), format="f32le", ac=2, ar=cfg.sample_rate)
        if np.isfinite(peak) and peak > 1.0:
            stream = stream.filter("volume", 1.0 / peak)
        try:
            stream.output(str(output_path), acodec="pcm_s16le").run(
                cmd="ffmpeg", capture_stdout=True, capture_stderr=True, overwrite_output=True
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to encode separated vocals: {e.stderr.decode()}") from e
    finally:
        raw_in.unlink(missing_ok=True)
        raw_out.unlink(missing_ok=True)
