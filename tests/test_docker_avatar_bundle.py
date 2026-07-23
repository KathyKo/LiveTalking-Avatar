from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_default_ditto_avatars_are_bundled_without_overwriting_volume():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    start = (ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for avatar in ("ditto_woman", "ditto_man"):
        assert f"COPY data/avatars/{avatar} /opt/default-avatars/{avatar}" in dockerfile
        assert f"!data/avatars/{avatar}/source.mp4" in ignore
        assert f"!data/avatars/{avatar}/idle.mp4" in ignore
    assert 'if ! compgen -G "$DATA_ROOT/avatars/$bundled_id/source.*"' in start
    assert 'cp -a "$bundled_avatar" "$DATA_ROOT/avatars/$bundled_id"' in start
