import dataclasses
import time
from io import StringIO
from threading import Thread, Semaphore
from typing import BinaryIO, Union

from faster_whisper import WhisperModel
from whisper.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG


def to_whisper_word(word):
    word["confidence"] = word.pop("probability")
    return word


class FasterWhisperASR(ASRModel):
    def __init__(self):
        super().__init__()
        self._slots = Semaphore(max(1, CONFIG.ASR_CONCURRENCY))

    def load_model(self):
        self.model = WhisperModel(
            model_size_or_path=CONFIG.MODEL_NAME,
            device=CONFIG.DEVICE,
            compute_type=CONFIG.MODEL_QUANTIZATION,
            download_root=CONFIG.MODEL_PATH,
            num_workers=max(1, CONFIG.CT2_NUM_WORKERS),
        )
        Thread(target=self.monitor_idleness, daemon=True).start()

    def transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
        output,
    ):
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None:
                self.load_model()

        options_dict = {"task": task}
        if language:
            options_dict["language"] = language
        if initial_prompt:
            options_dict["initial_prompt"] = initial_prompt
        if vad_filter:
            options_dict["vad_filter"] = True
        if word_timestamps:
            options_dict["word_timestamps"] = True
        with self._slots:
            segments = []
            text = ""
            segment_generator, info = self.model.transcribe(audio, beam_size=5, **options_dict)
            for segment in segment_generator:
                seg_dict = dataclasses.asdict(segment)
                if "words" in seg_dict and seg_dict["words"]:
                    seg_dict["words"] = [to_whisper_word(word) for word in seg_dict["words"]]
                segments.append(seg_dict)
                text += segment.text
            result = {"language": options_dict.get("language", info.language), "segments": segments, "text": text}

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None:
                self.load_model()

        segments, info = self.model.transcribe(audio, beam_size=5)
        return info.language, info.language_probability

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        options = {"max_line_width": 1000, "max_line_count": 10, "highlight_words": False}
        if output == "srt":
            WriteSRT(ResultWriter).write_result(result, file=file, options=options)
        elif output == "vtt":
            WriteVTT(ResultWriter).write_result(result, file=file, options=options)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file, options=options)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file, options=options)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file, options=options)
