"""
Mock heavy ML/C-extension dependencies before any app imports so tests
can run without torch, whisper, faster-whisper, etc. installed locally.
"""
import io
import sys
from unittest.mock import MagicMock

import pytest

# ------------------------------------------------------------------
# Install module stubs.  Every entry must be in sys.modules BEFORE
# any app.* import happens, which pytest guarantees because conftest
# is evaluated first.
# ------------------------------------------------------------------
_STUB_MODULES = [
    "torch",
    "whisper",
    "whisper.tokenizer",
    "whisper.utils",
    "faster_whisper",
    "faster_whisper.utils",
    "whisperx",
    "whisperx.audio",
    "whisperx.diarize",
    "whisperx.utils",
    "newrelic",
    "newrelic.agent",
    "ffmpeg",
    "librosa",
    "tqdm",
    "numba",
    "llvmlite",
    "soundfile",
    "huggingface_hub",
]

for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Provide realistic values the app code reads at import time.
sys.modules["torch"].cuda.is_available.return_value = False
sys.modules["whisper"].tokenizer = MagicMock()
sys.modules["whisper"].tokenizer.LANGUAGES = {"en": "english", "es": "spanish", "fr": "french"}
sys.modules["whisper.tokenizer"].LANGUAGES = {"en": "english", "es": "spanish", "fr": "french"}
for _cls in ["ResultWriter", "WriteJSON", "WriteSRT", "WriteTSV", "WriteTXT", "WriteVTT"]:
    setattr(sys.modules["whisper.utils"], _cls, MagicMock)
sys.modules["whisperx.audio"].N_SAMPLES = 48000
for _cls in ["ResultWriter", "SubtitlesWriter", "WriteJSON", "WriteSRT", "WriteTSV", "WriteTXT", "WriteVTT"]:
    setattr(sys.modules["whisperx.utils"], _cls, MagicMock)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_asr_model():
    """A MagicMock that behaves like a loaded ASR model."""
    model = MagicMock()
    model.transcribe.return_value = io.StringIO("hello world")
    model.language_detection.return_value = ("en", 0.95)
    return model


@pytest.fixture
def client(mock_asr_model):
    """
    FastAPI TestClient with the ASR model fully mocked.

    We reload app.webservice inside the patch context so the module-level
    ``asr_model = ASRModelFactory.create_asr_model()`` picks up the mock.
    """
    import importlib
    from unittest.mock import patch

    # app.webservice is likely already imported (module cache); reload it so
    # the module-level asr_model is re-created with the patched factory.
    with patch(
        "app.factory.asr_model_factory.ASRModelFactory.create_asr_model",
        return_value=mock_asr_model,
    ):
        import app.webservice as ws

        importlib.reload(ws)

        from fastapi.testclient import TestClient

        with TestClient(ws.app) as c:
            yield c, mock_asr_model
