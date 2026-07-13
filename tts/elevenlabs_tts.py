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
        self._voice_id = opt.REF_FILE or "DODLEQrClDo8wCz460ld"

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t = time.time()
        try:
            chunks = self._client.text_to_speech.stream(
                voice_id=self._voice_id,
                text=text,
                model_id="eleven_flash_v2_5",
                output_format="pcm_16000",
            )
            raw = b"".join(chunks)
        except Exception:
            logger.exception("elevenlabs tts error")
            return

        logger.info(f"elevenlabs tts time: {time.time()-t:.4f}s")
        stream = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            streamlen -= self.chunk
            if idx == 0:
                eventpoint = {"status": "start", "text": text}
            elif streamlen < self.chunk:
                eventpoint = {"status": "end", "text": text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk
