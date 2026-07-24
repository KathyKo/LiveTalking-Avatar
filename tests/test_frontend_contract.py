from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_avatar_frontend_contract():
    html = (ROOT / "web" / "index-en.html").read_text(encoding="utf-8")
    assert all(metric in html for metric in (
        "First character:", "First TTS:", "Complete response:", "Avatar started:"
    ))
    assert 'id="latencyFirst"' not in html
    assert "template.innerHTML = DOMPurify.sanitize(marked.parse(markdown))" in html
    assert "className = 'chat-media'" in html
    assert "(?:mp4|mov|webm)" in html
    assert "className = 'tour-qr'" in html
    assert "api.qrserver.com" in html
    assert "content.replaceChildren()" in html
    assert "s.startsWith('https://', i)" in html
    assert "row.hidden = true" in html
    assert "el.parentElement.hidden = false" in html
    assert "const interruptPromise = interrupt()" in html
    assert "fetch('/avatar_timing'" in html
    assert '<audio id="audio"' not in html
    assert "stream.addTrack(evt.track)" in html
    assert "aspect-ratio:4/3" in html
    assert "if (connecting || (pc && ['new', 'connecting', 'connected'].includes(pc.connectionState))) return;" in html
    assert "if (pc !== peer) return;" in html
    assert "scheduleReconnect(5000)" in html
    assert "function semanticBoundary(s, force = false)" in html
    assert "const HARD_MAX = 240;" in html
    assert "const CONTINUATION_PAUSE_MS = 80;" in html
    assert "paragraph ? 900 : 520" in html
    assert "pauseMs: 760" in html
    assert "'excl'" in html
    assert "pause_ms: pauseMs" in html
    assert "final, sessionid" in html
    assert "queueSpeak('', false, 20, true)" in html


def test_ditto_defaults_to_eight_steps_with_a_visible_tail_hold():
    script = (ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    assert "DITTO_STEPS=${DITTO_STEPS:-8}" in script
    assert "DITTO_FEED_CAP=${DITTO_FEED_CAP:-20}" in script
    assert "DITTO_START_BUFFER=${DITTO_START_BUFFER:-8}" in script
    assert "DITTO_HOLD=${DITTO_HOLD:-0.10}" in script
    assert "DITTO_TAIL_MS=${DITTO_TAIL_MS:-300}" in script
    assert "DITTO_IDLE_FADE_MS" not in script
    assert "DITTO_AV_OFFSET_MS=${DITTO_AV_OFFSET_MS:-0}" in script


def test_ditto_exposes_timing_events():
    avatar = (ROOT / "avatars" / "ditto_avatar.py").read_text(encoding="utf-8")
    routes = (ROOT / "server" / "routes.py").read_text(encoding="utf-8")
    assert "self._tts_start_seq += 1" in avatar
    assert "self._avatar_start_seq += 1" in avatar
    assert 'app.router.add_post("/avatar_timing", avatar_timing)' in routes


def test_elevenlabs_forwards_pcm_while_streaming():
    tts = (ROOT / "tts" / "elevenlabs_tts.py").read_text(encoding="utf-8")
    assert 'raw = b"".join(chunks)' not in tts
    assert "for pcm_chunk in chunks:" in tts
    assert "self.parent.put_audio_frame(frame, eventpoint)" in tts
    assert "re.split" not in tts
    assert "for _ in range(3)" not in tts
    assert "elevenlabs first audio:" in tts
    assert "previous_text=self._previous_text or None" in tts
    assert "for index in range((pause_ms + 19) // 20):" in tts
    assert '"segment_end"' in tts
