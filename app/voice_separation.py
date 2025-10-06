import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile
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

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VocalSeparationConfig":
        required = ["name", "sample_rate", "n_fft", "hop", "window", "dim_c", "dim_f", "dim_t"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        return VocalSeparationConfig(
            name=str(d["name"]),
            sample_rate=int(d["sample_rate"]),
            n_fft=int(d["n_fft"]),
            hop=int(d["hop"]),
            window=str(d["window"]),
            dim_c=int(d["dim_c"]),
            dim_f=int(d["dim_f"]),
            dim_t=int(d["dim_t"]),
        )

    def as_separate_kwargs(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "dim_f": self.dim_f,
        }


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
    precision: str = "fp32",
) -> np.ndarray:
    """
    Hacked from https://github.com/seanghay/vocal/blob/main/vocal/__init__.py.
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

    # Convert mix to torch tensor and move to GPU
    input_audio = torch.from_numpy(input_audio).to(device)

    margin = sample_rate if sample_rate < audio_chunk_size else audio_chunk_size
    samples = input_audio.shape[-1]

    if chunks == 0 or samples < audio_chunk_size:
        audio_chunk_size = samples

    # Pre-allocate chunks on GPU
    chunk_samples = []
    for skip in range(0, samples, audio_chunk_size):
        s_margin = 0 if skip == 0 else margin
        end = min(skip + audio_chunk_size + margin, samples)
        start = skip - s_margin
        chunk_samples.append(input_audio[:, start:end])
        if end == samples:
            break

    margin_size = margin
    chunked_sources = []

    amp_enabled = is_cuda and precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else (torch.bfloat16 if precision == "bf16" else None)

    for chunk_mix_position, chunk_mix in enumerate(chunk_samples):
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
            end = None if chunk_mix_position == len(chunk_samples) - 1 else -margin_size
            if margin_size == 0:
                end = None

        chunked_sources.append([tar_signal[:, start:end]])

    return np.concatenate(chunked_sources, axis=-1)[0]


def separate_vocals_from_array(
    audio: np.ndarray,
    model_id: str = "UVR-MDX-NET-Inst_HQ_4",
    repo_id: str = "mediainbox/uvr-mdx-models",
    device: str | None = None,
    chunks: int = 30,
    use_tta: bool = False,
    precision: str = "fp32",
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


def separate_vocals_from_file(
    input_path: str | Path,
    output_path: str | Path,
    model_id: str = "UVR-MDX-NET-Inst_HQ_4",
    repo_id: str = "mediainbox/uvr-mdx-models",
    device: str | None = None,
    chunks: int = 30,
    use_tta: bool = False,
    precision: str = "fp32",
) -> None:
    device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()

    input_path = str(input_path)
    output_path = str(output_path)

    # 1. Load model and config
    model, cfg = get_model_and_config(
        model_id=model_id,
        repo_id=repo_id,
        device=device,
    )

    # 2. Read audio accordingly
    audio, _ = librosa.load(input_path, sr=cfg.sample_rate, mono=False)

    # 3. Separate vocals
    audio_vocals = separate_vocal(
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
    )

    # 4. Save vocals
    soundfile.write(output_path, audio_vocals.T, samplerate=cfg.sample_rate)
