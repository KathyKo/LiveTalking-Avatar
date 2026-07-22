from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_avatar_frontend_contract():
    html = (ROOT / "web" / "index-en.html").read_text(encoding="utf-8")
    assert all(metric in html for metric in (
        'id="latencyFirst"', 'id="latencyComplete"', 'id="latencyAvatar"'
    ))
    assert "DOMPurify.sanitize(marked.parse(markdown))" in html
    assert "className = 'chat-media'" in html
    assert '<audio id="audio"' not in html
    assert "stream.addTrack(evt.track)" in html
    assert "aspect-ratio:4/3" in html
    assert "if (connecting || (pc && ['new', 'connecting', 'connected'].includes(pc.connectionState))) return;" in html
    assert "if (pc !== peer) return;" in html
    assert "scheduleReconnect(5000)" in html


def test_ditto_defaults_to_ten_steps():
    script = (ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    assert "DITTO_STEPS=${DITTO_STEPS:-10}" in script
