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
    assert "const CHAT_AGENT_ID = 'JTC_M';" in html
    assert "agent_id: CHAT_AGENT_ID" in html
    assert "fetch('/avatar_timing'" in html
    assert '<audio id="audio"' not in html
    assert "stream.addTrack(evt.track)" in html
    assert "html, body {" in html
    assert "overflow: hidden;" in html
    assert "height: 100dvh" in html
    assert "grid-template-columns: minmax(0, 1fr) minmax(380px, 1fr)" in html
    assert "#chatResponse { flex: 1;" in html
    assert "overflow-y: auto !important" in html
    assert 'id="btnMic" type="button" onclick="toggleMic()"' in html
    assert html.count('id="btnMic"') == 1
    assert "Press to Talk" not in html
    assert 'onclick="sendText()"' not in html
    assert "function renderMicButton(listening)" in html
    assert ".chat-bubble { max-width: 94%;" in html
    assert "if (connecting || (pc && ['new', 'connecting', 'connected'].includes(pc.connectionState))) return;" in html
    assert "if (pc !== peer) return;" in html
    assert "scheduleReconnect(5000)" in html
    assert "function semanticBoundary(s, force = false)" in html
    semantic = html.split("function semanticBoundary", 1)[1].split("function addBubble", 1)[0]
    assert "HARD_MAX" not in semantic
    assert "lastSpace" not in semantic
    assert "s.startsWith(' - ', i)" in semantic
    assert "pauseMs: 280" in semantic
    assert "paragraph ? 900 : 520" in html
    assert "pauseMs: 760" in html
    assert "'excl'" in html
    assert "pause_ms: pauseMs" in html
    assert "final, sessionid" in html
    assert "queueSpeak('', false, 20, true)" in html


def test_ditto_defaults_to_fast_high_resolution_rendering():
    script = (ROOT / "docker" / "ditto-env.sh").read_text(encoding="utf-8")
    # Sourced, not duplicated: a shell in JupyterLab must be able to reproduce
    # the server's exact rendering parameters before regenerating idle.mp4.
    start = (ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    assert 'source "$APP_ROOT/docker/ditto-env.sh"' in start
    assert "DITTO_STEPS" not in start, "defaults must live in ditto-env.sh only"
    assert "DITTO_STEPS=${DITTO_STEPS:-5}" in script
    assert "DITTO_EXP=${DITTO_EXP:-0.75}" in script
    assert "DITTO_MAX_SIZE=${DITTO_MAX_SIZE:-896}" in script
    assert "DITTO_LIP_RESPONSE=${DITTO_LIP_RESPONSE:-1.15}" in script
    assert "DITTO_FEED_CAP=${DITTO_FEED_CAP:-20}" in script
    assert "DITTO_START_BUFFER=${DITTO_START_BUFFER:-6}" in script
    assert "DITTO_HOLD=${DITTO_HOLD:-0.10}" in script
    assert "DITTO_TAIL_MS=${DITTO_TAIL_MS:-500}" in script
    assert "DITTO_IDLE_FADE_MS" not in script
    assert "DITTO_AV_OFFSET_MS=${DITTO_AV_OFFSET_MS:-260}" in script
    assert "DITTO_FINAL_HOLD_MS=${DITTO_FINAL_HOLD_MS:-220}" in script
    assert "DITTO_GENERATE_IDLE=${DITTO_GENERATE_IDLE:-0}" in script
    assert "DITTO_USE_GENERATED_IDLE=${DITTO_USE_GENERATED_IDLE:-0}" in script


def test_generated_idle_is_optional_and_preserves_original():
    start = (ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    avatar = (ROOT / "avatars" / "ditto_avatar.py").read_text(encoding="utf-8")
    generator = (ROOT / "scripts" / "ditto_make_idle.py").read_text(encoding="utf-8")

    assert 'if [[ "$DITTO_GENERATE_IDLE" == "1" ]]' in start
    assert "WARNING: generated idle failed; using original idle.mp4" in start
    assert 'os.environ.get("DITTO_USE_GENERATED_IDLE", "0") == "1"' in avatar
    assert '("idle.generated.mp4", "idle.mp4") if use_generated else ("idle.mp4",)' in avatar
    assert 'not os.path.exists(idle_path + ".json")' in avatar
    assert "neutralize_sdk_source_lips(sdk, original_idle)" in generator
    assert 'sdk.setup(original_idle, "/tmp/ditto_make_idle_dummy.mp4"' in generator
    assert "GENERATOR_VERSION = 3" in generator
    assert 'original_idle = os.path.join(avatar_dir, "idle.mp4")' in generator
    assert "os.replace(temporary_out, out)" in generator
    assert "os.replace(temporary_out, original_idle)" not in generator


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
    # Frames leave while the stream is still arriving. Only the segment's first
    # _ESTIMATE_FRAMES are held back, to fix its loudness against other segments.
    assert "self._emit([frame], text, textevent, gain)" in tts
    assert "np.clip(frame * gain, -1.0, 1.0), eventpoint)" in tts
    assert "if len(held) >= _ESTIMATE_FRAMES:" in tts
    assert "re.split" not in tts
    assert "for _ in range(3)" not in tts
    assert "elevenlabs first audio:" in tts
    assert "previous_text=self._previous_text or None" in tts
    assert "for index in range((pause_ms + 19) // 20):" in tts
    assert '"segment_end"' in tts
