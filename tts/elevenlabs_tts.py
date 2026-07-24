import os
import time
import numpy as np
from elevenlabs.client import ElevenLabs

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register


@register("tts", "elevenlabs")
class ElevenLabsTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self._client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        self._voice_id = opt.REF_FILE or "SEWXl8lPSO01tdGbWECX"
        self._previous_text = ""

    def flush_talk(self):
        super().flush_talk()
        self._previous_text = ""

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        text = text.strip()
        final = bool(textevent.get("final"))
        if not text:
            if final:
                self._send_silence_tail(text, textevent, True)
            return

        first = True
        started = time.perf_counter()
        try:
            chunks = self._client.text_to_speech.stream(
                voice_id=self._voice_id,
                text=text,
                model_id="eleven_flash_v2_5",
                output_format="pcm_16000",
                previous_text=self._previous_text or None,
            )
            pcm_buffer = bytearray()
            frame_bytes = self.chunk * np.dtype(np.int16).itemsize
            got_audio = False
            for pcm_chunk in chunks:
                if self.state != State.RUNNING:
                    return
                if not got_audio:
                    got_audio = True
                    logger.info("elevenlabs first audio: %.4fs", time.perf_counter() - started)
                pcm_buffer.extend(pcm_chunk)
                while len(pcm_buffer) >= frame_bytes:
                    raw_frame = bytes(pcm_buffer[:frame_bytes])
                    del pcm_buffer[:frame_bytes]
                    frame = np.frombuffer(raw_frame, dtype=np.int16).astype(np.float32) / 32768.0
                    eventpoint = {}
                    if first:
                        eventpoint = {"status": "start", "text": text}
                        first = False
                    eventpoint.update(**textevent)
                    self.parent.put_audio_frame(frame, eventpoint)
        except Exception:
            logger.exception("elevenlabs tts error")
            return

        logger.info("elevenlabs stream/feed complete: %.4fs", time.perf_counter() - started)
        usable_bytes = len(pcm_buffer) - (len(pcm_buffer) % 2)
        if usable_bytes:
            frame = np.frombuffer(bytes(pcm_buffer[:usable_bytes]), dtype=np.int16).astype(np.float32) / 32768.0
            frame = np.pad(frame, (0, self.chunk - len(frame)))
            eventpoint = {}
            if first:
                eventpoint = {"status": "start", "text": text}
                first = False
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(frame, eventpoint)

        self._send_silence_tail(text, textevent, final)
        self._previous_text = text

    def _send_silence_tail(self, text, textevent, final):
        # Sentence pauses keep the mouth closed without resetting Ditto's audio
        # window. Only the final marker flushes the window and returns to idle.
        pause_ms = max(20, int(textevent.get("pause_ms", os.environ.get("DITTO_TAIL_MS", "300"))))
        for index in range((pause_ms + 19) // 20):
            eventpoint = {}
            if index * 20 + 20 >= pause_ms:
                eventpoint = {"status": "end" if final else "segment_end", "text": text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
