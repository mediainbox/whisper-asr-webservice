import os

import torch


class CONFIG:
    """
    Configuration class for ASR models.
    Reads environment variables for runtime configuration, with sensible defaults.
    """
    # Determine the ASR engine ('faster_whisper', 'openai_whisper' or 'whisperx')
    ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")

    # Retrieve Huggingface Token
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    if ASR_ENGINE == "whisperx" and HF_TOKEN == "":
        print("You must set the HF_TOKEN environment variable to download the diarization model used by WhisperX.")

    # Determine the computation device (GPU or CPU)
    DEVICE = os.getenv("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Model name to use (e.g., "base", "small", etc.)
    MODEL_NAME = os.getenv("ASR_MODEL", "base")

    # Path to the model directory
    MODEL_PATH = os.getenv("ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))

    # Model quantization level. Defines the precision for model weights:
    #   'float32' - 32-bit floating-point precision (higher precision, slower inference)
    #   'float16' - 16-bit floating-point precision (lower precision, faster inference)
    #   'int8' - 8-bit integer precision (lowest precision, fastest inference)
    # Defaults to 'float32' for GPU availability, 'int8' for CPU.
    MODEL_QUANTIZATION = os.getenv("ASR_QUANTIZATION", "float32" if torch.cuda.is_available() else "int8")
    if MODEL_QUANTIZATION not in {"float32", "float16", "int8"}:
        raise ValueError("Invalid MODEL_QUANTIZATION. Choose 'float32', 'float16', or 'int8'.")

    # Idle timeout in seconds. If set to a non-zero value, the model will be unloaded
    # after being idle for this many seconds. A value of 0 means the model will never be unloaded.
    MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", 0))

    # Default sample rate for audio input. 16 kHz is commonly used in speech-to-text tasks.
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))

    # Subtitle output options for whisperx
    SUBTITLE_MAX_LINE_WIDTH = int(os.getenv("SUBTITLE_MAX_LINE_WIDTH", 1000))
    SUBTITLE_MAX_LINE_COUNT = int(os.getenv("SUBTITLE_MAX_LINE_COUNT", 2))
    SUBTITLE_HIGHLIGHT_WORDS = os.getenv("SUBTITLE_HIGHLIGHT_WORDS", "false").lower() == "true"

    # Voice separation options
    VOICE_SEPARATION_MODEL = os.getenv("VOICE_SEPARATION_MODEL", "UVR-MDX-NET-Inst_HQ_4")
    VOICE_SEPARATION_PRECISION = os.getenv("VOICE_SEPARATION_PRECISION", "fp16")

    # Max concurrent GPU operations (legacy fallback — sets both semaphores if the
    # specific vars below are not set). Keep at 1 if OOM errors appear.
    GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", 1))

    # Concurrent vocal separation operations. UVR-MDX-NET is heavier than transcription;
    # lower this if VRAM is tight. Defaults to GPU_CONCURRENCY for backwards compat.
    VOCALS_CONCURRENCY = int(os.getenv("VOCALS_CONCURRENCY", os.getenv("GPU_CONCURRENCY", 1)))

    # Concurrent transcription operations. faster-whisper is lighter than vocal separation;
    # can be set higher than VOCALS_CONCURRENCY to improve pipeline throughput.
    TRANSCRIBE_CONCURRENCY = int(os.getenv("TRANSCRIBE_CONCURRENCY", os.getenv("GPU_CONCURRENCY", 1)))

    # Concurrent audio decode/preprocess operations. This is CPU + host-RAM bound, NOT GPU,
    # and it is what drives host-RAM OOMs: each decode holds the whole clip as a float32
    # numpy array (~230 MB per hour of audio), plus transient copies during conversion.
    # It must be capped independently of the GPU semaphores — otherwise N concurrent uploads
    # decode N full clips into RAM at once even when TRANSCRIBE_CONCURRENCY is 1.
    # ponytail: fixed cap on concurrent decodes; raise per available host RAM, not per GPU.
    DECODE_CONCURRENCY = int(os.getenv("DECODE_CONCURRENCY", os.getenv("GPU_CONCURRENCY", 2)))

    # Max concurrent HTTP connections accepted by uvicorn. Requests beyond this limit
    # receive an immediate 503 at the transport layer — before any app code runs.
    # Prevents FD exhaustion under traffic spikes. 0 = unlimited (uvicorn default).
    UVICORN_LIMIT_CONCURRENCY = int(os.getenv("UVICORN_LIMIT_CONCURRENCY", 0)) or None
