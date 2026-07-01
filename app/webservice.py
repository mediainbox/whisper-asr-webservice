import asyncio
import importlib.metadata
import io
import os
import time
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
from app.voice_separation import load_audio_for_separation, run_separation_gpu, separate_vocals_from_file

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio, timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# Separate semaphores allow independent tuning of each GPU phase.
# Vocal separation (UVR-MDX-NET) is heavier; transcription (faster-whisper) is lighter
# and can run at higher concurrency without exhausting VRAM.
# Both default to GPU_CONCURRENCY for backwards compatibility.
_vocals_semaphore    = asyncio.Semaphore(CONFIG.VOCALS_CONCURRENCY)
_transcribe_semaphore = asyncio.Semaphore(CONFIG.TRANSCRIBE_CONCURRENCY)
# Caps host-RAM use: decode holds the full clip as a float32 array (~230 MB/hour).
# Without this gate, concurrent uploads decode all clips into RAM at once and OOM the host.
_decode_semaphore    = asyncio.Semaphore(CONFIG.DECODE_CONCURRENCY)

def _reject_if_full(sem: asyncio.Semaphore):
    """Reject with 503 right at the point of use instead of queuing indefinitely
    on `async with sem`. A single check at request entry is a stale snapshot —
    capacity can fill during upload I/O or a burst of concurrent admissions
    between the check and the actual acquire, and asyncio.Semaphore has no
    acquire timeout, so admitted-but-blocked requests pile up in _requests_active
    forever. Checking again immediately before each acquire closes that gap."""
    if sem._value == 0:
        from fastapi import Response
        return Response(status_code=503, headers={"Retry-After": "5"})
    return None


_start_time = time.time()
_requests_total = 0
_requests_active = 0

asr_model = ASRModelFactory.create_asr_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr_model.load_model()
    try:
        yield
    finally:
        asr_model.release_model()


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


@app.get("/health", tags=["Health"], include_in_schema=False)
async def health():
    if asr_model.model is None:
        from fastapi import Response
        return Response(status_code=503, content="Model not loaded")
    return {"status": "ok"}


def _os_metrics():
    """Best-effort host/process metrics. Linux /proc + stdlib only; returns whatever is
    available (so it degrades to load_avg-only on non-Linux dev machines). These surface
    the host-RAM OOM signals directly: memory.available_mib, swap, and process.rss_mib."""
    m = {}
    try:
        m["cpu_count"] = os.cpu_count()
        m["load_avg"] = [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        pass
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0])  # values in kB
        total = info["MemTotal"]
        available = info.get("MemAvailable", info.get("MemFree", 0))
        m["memory"] = {
            "total_mib": round(total / 1024),
            "available_mib": round(available / 1024),
            "used_mib": round((total - available) / 1024),
            "percent_used": round(100 * (total - available) / total, 1) if total else None,
            "swap_total_mib": round(info.get("SwapTotal", 0) / 1024),
            "swap_free_mib": round(info.get("SwapFree", 0) / 1024),
        }
    except (OSError, KeyError, ValueError):
        pass
    try:
        proc = {}
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmSize:")):
                    key, _, rest = line.partition(":")
                    proc[key] = round(int(rest.split()[0]) / 1024)  # kB -> MiB
        if proc:
            m["process"] = {"rss_mib": proc.get("VmRSS"), "vm_mib": proc.get("VmSize")}
    except (OSError, ValueError):
        pass
    return m


@app.get("/stats", tags=["Health"], include_in_schema=False)
async def stats():
    import torch
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            entry = {
                "index": i,
                "name": props.name,
                "memory_allocated_mib": round(torch.cuda.memory_allocated(i) / 1024 ** 2),
                "memory_reserved_mib": round(torch.cuda.memory_reserved(i) / 1024 ** 2),
                "memory_total_mib": round(props.total_memory / 1024 ** 2),
            }
            # Actual device memory (not just torch's caching allocator view) + util/temp.
            # ponytail: best-effort — util/temp need pynvml under the hood and may be absent.
            try:
                free, total = torch.cuda.mem_get_info(i)
                entry["memory_free_mib"] = round(free / 1024 ** 2)
                entry["memory_used_mib"] = round((total - free) / 1024 ** 2)
            except Exception:
                pass
            try:
                entry["utilization_percent"] = torch.cuda.utilization(i)
            except Exception:
                pass
            try:
                entry["temperature_c"] = torch.cuda.temperature(i)
            except Exception:
                pass
            gpu.append(entry)

    return {
        "uptime_seconds": round(time.time() - _start_time),
        "requests": {
            "total": _requests_total,
            "active": _requests_active,
        },
        "model": {
            "engine": CONFIG.ASR_ENGINE,
            "name": CONFIG.MODEL_NAME,
            "device": CONFIG.DEVICE,
            "quantization": CONFIG.MODEL_QUANTIZATION,
            "loaded": asr_model.model is not None,
            "idle_seconds": round(time.time() - asr_model.last_activity_time),
        },
        "decode": {
            "concurrency": CONFIG.DECODE_CONCURRENCY,
            "slots_used": CONFIG.DECODE_CONCURRENCY - _decode_semaphore._value,
            "slots_free": _decode_semaphore._value,
        },
        "transcribe": {
            "concurrency": CONFIG.TRANSCRIBE_CONCURRENCY,
            "slots_used": CONFIG.TRANSCRIBE_CONCURRENCY - _transcribe_semaphore._value,
            "slots_free": _transcribe_semaphore._value,
        },
        "vocals": {
            "concurrency": CONFIG.VOCALS_CONCURRENCY,
            "slots_used": CONFIG.VOCALS_CONCURRENCY - _vocals_semaphore._value,
            "slots_free": _vocals_semaphore._value,
        },
        "os": _os_metrics(),
        "gpu": gpu,
    }


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

    if _transcribe_semaphore._value == 0:
        from fastapi import Response
        return Response(status_code=503, headers={"Retry-After": "5"})
    if separate_vocals and _vocals_semaphore._value == 0:
        from fastapi import Response
        return Response(status_code=503, headers={"Retry-After": "5"})

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
        }.items()
    )

    global _requests_total, _requests_active
    _requests_total += 1
    _requests_active += 1

    if verbose:
        logger.info("Analyzing file: %s", filename)

    result = None

    if separate_vocals:
        suffix = Path(filename).suffix or ".wav"

        with (
            newrelic.agent.FunctionTrace(name="asr.upload_read_all", group="ASR"),
            timer("upload_read_all", enabled=verbose),
        ):
            async_bytes = await audio_file.read()

        with TemporaryDirectory() as td:
            td = Path(td)

            in_path, out_path = td / f"in{suffix}", td / "vocals.wav"

            with open(in_path, "wb") as f_in:
                f_in.write(async_bytes)

            with (
                newrelic.agent.FunctionTrace(name="asr.total_after_disk", group="ASR"),
                timer("total_after_disk", enabled=verbose),
            ):
                with (
                    newrelic.agent.FunctionTrace(name="asr.separate_vocals", group="ASR"),
                    timer("separate_vocals", enabled=verbose),
                ):
                    # CPU phase: decode audio and fetch cached model — no GPU needed.
                    # Decode is the RAM-heavy step, so it goes under the decode gate.
                    if rejected := _reject_if_full(_decode_semaphore):
                        _requests_active -= 1
                        return rejected
                    async with _decode_semaphore:
                        audio_raw, vs_cfg, vs_model, vs_device = await asyncio.to_thread(
                            load_audio_for_separation,
                            in_path,
                            model_id=CONFIG.VOICE_SEPARATION_MODEL,
                        )
                    # GPU phase: vocal separation inference
                    if rejected := _reject_if_full(_vocals_semaphore):
                        _requests_active -= 1
                        return rejected
                    async with _vocals_semaphore:
                        await asyncio.to_thread(
                            run_separation_gpu,
                            audio_raw,
                            vs_cfg,
                            vs_model,
                            vs_device,
                            out_path,
                            precision=CONFIG.VOICE_SEPARATION_PRECISION,
                        )

                with open(out_path, "rb") as f_vocals:
                    with (
                        newrelic.agent.FunctionTrace(name="asr.load_audio_vocals", group="ASR"),
                        timer("load_audio(vocals)", enabled=verbose),
                    ):
                        if rejected := _reject_if_full(_decode_semaphore):
                            _requests_active -= 1
                            return rejected
                        async with _decode_semaphore:
                            audio_np = await asyncio.to_thread(load_audio, f_vocals, encode=True)

                with (
                    newrelic.agent.FunctionTrace(name="asr.transcribe", group="ASR"),
                    timer(f"transcribe({CONFIG.ASR_ENGINE})", enabled=verbose),
                ):
                    if rejected := _reject_if_full(_transcribe_semaphore):
                        _requests_active -= 1
                        return rejected
                    async with _transcribe_semaphore:
                        result = await asyncio.to_thread(
                            asr_model.transcribe,
                            audio_np,
                            task,
                            language,
                            initial_prompt,
                            vad_filter,
                            word_timestamps,
                            {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
                            output,
                        )
    else:
        with (
            newrelic.agent.FunctionTrace(name="asr.load_audio_original", group="ASR"),
            timer("load_audio(original)", enabled=verbose),
        ):
            if rejected := _reject_if_full(_decode_semaphore):
                _requests_active -= 1
                return rejected
            async with _decode_semaphore:
                audio_np = await asyncio.to_thread(load_audio, audio_file.file, encode)

        with (
            newrelic.agent.FunctionTrace(name="asr.total_after_decode", group="ASR"),
            timer("total_after_decode", enabled=verbose),
        ):
            with (
                newrelic.agent.FunctionTrace(name="asr.transcribe", group="ASR"),
                timer(f"transcribe({CONFIG.ASR_ENGINE})", enabled=verbose),
            ):
                if rejected := _reject_if_full(_transcribe_semaphore):
                    _requests_active -= 1
                    return rejected
                async with _transcribe_semaphore:
                    result = await asyncio.to_thread(
                        asr_model.transcribe,
                        audio_np,
                        task,
                        language,
                        initial_prompt,
                        vad_filter,
                        word_timestamps,
                        {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
                        output,
                    )

    _requests_active -= 1

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
            async with _vocals_semaphore:
                await asyncio.to_thread(
                    separate_vocals_from_file,
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


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    audio_np = await asyncio.to_thread(load_audio, audio_file.file, encode)
    detected_lang_code, confidence = await asyncio.to_thread(asr_model.language_detection, audio_np)
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
    uvicorn.run("app.webservice:app", host=host, port=port, workers=workers,
                limit_concurrency=CONFIG.UVICORN_LIMIT_CONCURRENCY)


if __name__ == "__main__":
    start()
