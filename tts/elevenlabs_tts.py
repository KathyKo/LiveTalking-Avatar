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
                raw = b"".join(chunks)
            except Exception:
                logger.exception("elevenlabs tts error")
                return

            logger.info(f"elevenlabs tts time: {time.time()-t_part:.4f}s")
            stream = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            streamlen = stream.shape[0]
            idx = 0
            while idx < streamlen:
                frame = stream[idx:idx + self.chunk]
                if len(frame) < self.chunk:
                    frame = np.pad(frame, (0, self.chunk - len(frame)))
                eventpoint = {}
                if first:
                    eventpoint = {"status": "start", "text": text}
                    first = False
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(frame, eventpoint)
                idx += self.chunk

        for _ in range(3):
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), {})
        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
