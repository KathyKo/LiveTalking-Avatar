from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_avatar_frontend_contract():
    html = (ROOT / "web" / "index-en.html").read_text(encoding="utf-8")
    assert all(metric in html for metric in (
        "First character:", "First TTS:", "Complete response:", "Avatar started:"
    ))
    assert 'id="latencyFirst"' not in html
    assert "DOMPurify.sanitize(marked.parse(markdown))" in html
    assert "className = 'chat-media'" in html
    assert "(?:mp4|mov|webm)" in html
    assert "fetch('/avatar_timing'" in html
    assert '<audio id="audio"' not in html
    assert "stream.addTrack(evt.track)" in html
    assert "aspect-ratio:4/3" in html
    assert "if (connecting || (pc && ['new', 'connecting', 'connected'].includes(pc.connectionState))) return;" in html
    assert "if (pc !== peer) return;" in html
    assert "scheduleReconnect(5000)" in html


def test_ditto_defaults_to_ten_steps():
    script = (ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    assert "DITTO_STEPS=${DITTO_STEPS:-10}" in script


def test_ditto_exposes_timing_events():
    avatar = (ROOT / "avatars" / "ditto_avatar.py").read_text(encoding="utf-8")
    routes = (ROOT / "server" / "routes.py").read_text(encoding="utf-8")
    assert "self._tts_start_seq += 1" in avatar
    assert "self._avatar_start_seq += 1" in avatar
    assert 'app.router.add_post("/avatar_timing", avatar_timing)' in routes
