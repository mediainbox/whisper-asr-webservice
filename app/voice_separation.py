import json
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ModelConfig:
    name: str
    sample_rate: int
    n_fft: int
    hop: int
    window: str
    dim_c: int
    dim_f: int
    dim_t: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ModelConfig":
        required = ["name", "sample_rate", "n_fft", "hop", "window", "dim_c", "dim_f", "dim_t"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        return ModelConfig(
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


def load_uvr_model_and_config(
    model_id: str = "UVR-MDX-NET-Voc_FT",
    repo_id: str = "mediainbox/uvr-mdx-models",
    device: str = "cuda",
):
    device = device.lower()
    model_path = hf_hub_download(repo_id, f"{model_id}.pt")
    cfg_path = hf_hub_download(repo_id, f"{model_id}.json")

    model = torch.jit.load(model_path, map_location=device)
    with open(cfg_path, "r") as f:
        cfg = ModelConfig.from_dict(json.load(f))

    return model, cfg
