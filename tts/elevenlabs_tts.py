import os
import re
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

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t = time.time()
        try:
            parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text.strip()) if p.strip()]
            if not parts:
                return
        except Exception:
            logger.exception("elevenlabs tts error")
            return

        first = True
        for part in parts:
            if self.state != State.RUNNING:
                return
            t_part = time.time()
            try:
                chunks = self._client.text_to_speech.stream(
                    voice_id=self._voice_id,
                    text=part,
                    model_id="eleven_flash_v2_5",
                    output_format="pcm_16000",
                )
                pcm_buffer = bytearray()
                frame_bytes = self.chunk * np.dtype(np.int16).itemsize
                for pcm_chunk in chunks:
                    if self.state != State.RUNNING:
                        return
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

            logger.info(f"elevenlabs tts time: {time.time()-t_part:.4f}s")
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

        for _ in range(3):
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), {})
        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
