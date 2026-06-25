"""
Voncierge integration — replaces llm.py for the LiveTalking /human chat endpoint.
Calls the Voncierge LangGraph agent instead of llm.
"""
import re
import time
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from llm import clean_text
from utils.logger import logger

VONCIERGE_URL = "http://127.0.0.1:8000/chat"


def voncierge_response(message: str, avatar_session: "BaseAvatar", datainfo: dict = {}):
    try:
        sessionid = getattr(avatar_session.opt, 'sessionid', '0')

        start = time.perf_counter()
        resp = requests.post(VONCIERGE_URL, json={
            "message": message,
            "session_id": f"livetalk_{sessionid}"
        }, timeout=15)
        resp.raise_for_status()
        reply = resp.json()["reply"]
        logger.info(f"[Voncierge] Reply in {time.perf_counter()-start:.2f}s: {reply[:100]}")

        # Split on sentence boundaries so avatar starts speaking sentence by sentence
        sentences = re.split(r'(?<=[.!?])\s+', clean_text(reply))
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                avatar_session.put_msg_txt(sentence, datainfo)

    except Exception:
        logger.exception("[Voncierge] Error:")
