import importlib.metadata
import io
import os
from contextlib import asynccontextmanager
from os import path
from pathlib import Path
from typing import Annotated, Union
from urllib.parse import quote
import logging

import click
import newrelic.agent
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer
from tempfile import TemporaryDirectory
from app.voice_separation import separate_vocals_from_file

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio, timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

asr_model = ASRModelFactory.create_asr_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr_model.load_model()
    yield


app = FastAPI(
    lifespan=lifespan,
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={
        "name": "MIT License",
        "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE",
    },
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@newrelic.agent.web_transaction(name="POST /asr", group="ASR")
@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    separate_vocals: Annotated[
        bool | None,
        Query(
            description="Preprocess with voice separation (fast-whisper only)",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    verbose: bool = Query(default=False, description="Print timing info for preprocessing and transcription"),
):
    filename = getattr(audio_file, "filename", "") or "audio.wav"

    newrelic.agent.add_custom_attributes(
        {
            "asr.engine": CONFIG.ASR_ENGINE,
            "asr.model": CONFIG.MODEL_NAME,
            "asr.device": CONFIG.DEVICE,
            "asr.compute_type": CONFIG.MODEL_QUANTIZATION,
            "asr.encode": bool(encode),
            "asr.task": task or "",
            "asr.language": language or "",
            "asr.vad_filter": bool(vad_filter),
            "asr.word_timestamps": bool(word_timestamps),
            "asr.separate_vocals": bool(separate_vocals),
            "asr.output": output or "",
        }
    )

    if verbose:
        logger.info("Analyzing file: %s", filename)

    if separate_vocals:
        suffix = Path(filename).suffix or ".wav"
        async_bytes = await audio_file.read()

        with TemporaryDirectory() as td:
            td = Path(td)

            in_path, out_path = td / f"in{suffix}", td / "vocals.wav"

            with open(in_path, "wb") as f_in:
                f_in.write(async_bytes)

            with (
                newrelic.agent.FunctionTrace(name="asr.separate_vocals", group="ASR"),
                timer("separate_vocals", enabled=verbose),
            ):
                separate_vocals_from_file(
                    in_path,
                    out_path,
                    model_id=CONFIG.VOICE_SEPARATION_MODEL,
                    precision=CONFIG.VOICE_SEPARATION_PRECISION,
                )

            with open(out_path, "rb") as f_vocals:
                with (
                    newrelic.agent.FunctionTrace(name="asr.load_audio_vocals", group="ASR"),
                    timer("load_audio(vocals)", enabled=verbose),
                ):
                    audio_np = load_audio(f_vocals, encode=True)
    else:
        with (
            newrelic.agent.FunctionTrace(name="asr.load_audio_original", group="ASR"),
            timer("load_audio(original)", enabled=verbose),
        ):
            audio_np = load_audio(audio_file.file, encode)

    with (
        newrelic.agent.FunctionTrace(name="asr.transcribe", group="ASR"),
        timer(f"transcribe({CONFIG.ASR_ENGINE})", enabled=verbose),
    ):
        result = asr_model.transcribe(
            audio_np,
            task,
            language,
            initial_prompt,
            vad_filter,
            word_timestamps,
            {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
            output,
        )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(filename)}.{output}"',
        },
    )


@app.post("/separate-vocals", tags=["Endpoints"])
async def separate_vocals(
    audio_file: UploadFile = File(...),  # noqa: B008
    verbose: bool = Query(default=False, description="Print timing info for voice separation"),
):
    filename = getattr(audio_file, "filename", "") or "audio.wav"

    if verbose:
        logger.info("Analyzing file: %s", filename)

    suffix = Path(filename).suffix or ".wav"
    stem = Path(filename).stem or "audio"

    async_bytes = await audio_file.read()
    with TemporaryDirectory() as td:
        td = Path(td)
        in_path = td / f"in{suffix}"
        out_path = td / "vocals.wav"

        with open(in_path, "wb") as f_in:
            f_in.write(async_bytes)

        with timer("separate_vocals", enabled=verbose):
            separate_vocals_from_file(
                in_path,
                out_path,
                model_id=CONFIG.VOICE_SEPARATION_MODEL,
                precision=CONFIG.VOICE_SEPARATION_PRECISION,
            )

        with open(out_path, "rb") as f_vocals:
            data = f_vocals.read()

    download_name = f"{stem}_vocals.wav"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{quote(download_name)}"',
        },
    )


@newrelic.agent.web_transaction(name="POST /detect-language", group="ASR")
@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code, confidence = asr_model.language_detection(load_audio(audio_file.file, encode))
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    show_default=True,
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    type=int,
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.option(
    "-w",
    "--workers",
    type=click.IntRange(1),
    default=1,
    show_default=True,
    help="Number of worker processes",
)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: int, workers: int):
    uvicorn.run("app.webservice:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    start()
