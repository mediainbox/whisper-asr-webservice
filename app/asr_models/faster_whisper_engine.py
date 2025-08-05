import time
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union

import whisper
from faster_whisper import WhisperModel
from whisper.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG


def to_whisper_word(word):
    word_dict = word._asdict()
    word_dict["confidence"] = word_dict.pop("probability")
    return word_dict


class FasterWhisperASR(ASRModel):

    def load_model(self):

        self.model = WhisperModel(
            model_size_or_path=CONFIG.MODEL_NAME,
            device=CONFIG.DEVICE,
            compute_type=CONFIG.MODEL_QUANTIZATION,
            download_root=CONFIG.MODEL_PATH
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
        with self.model_lock:
            segments = []
            text = ""
            segment_generator, info = self.model.transcribe(audio, beam_size=5, **options_dict)
            for segment in segment_generator:
                seg_dict = segment._asdict()
                if "words" in seg_dict and seg_dict["words"]:
                    seg_dict["words"] = [to_whisper_word(word) for word in seg_dict["words"]]
                segments.append(seg_dict)
                text = text + segment.text
            result = {"language": options_dict.get("language", info.language), "segments": segments, "text": text}

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):

        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None: self.load_model()

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)

        # detect the spoken language
        with self.model_lock:
            segments, info = self.model.transcribe(audio, beam_size=5)
            detected_lang_code = info.language
            detected_language_confidence = info.language_probability

        return detected_lang_code, detected_language_confidence

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
