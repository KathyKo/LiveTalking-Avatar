import ast
from pathlib import Path


def test_tail_batches_match_audio_duration():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tail_frame_counts"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<tail>", "exec"), namespace)
    plan = namespace["_tail_frame_counts"]

    assert plan(2, 0) == [1]
    assert plan(50, 20) == [5]
    assert plan(52, 20) == [5, 1]
    assert plan(50, 25) == []

    assert "self._frame_keep.put(i < keep_frames)" in source
    assert "if not keep_frame:" in source
    assert "if in_speech:" in source
    assert "return not self._audio_out.empty()" in source
    assert 'DITTO_IDLE_FADE_MS", "120"' in source
    assert 'DITTO_AV_OFFSET_MS", "0"' in source


def test_tts_silence_tail_marks_only_its_final_frame():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    assert "for index in range((pause_ms + 19) // 20):" in source
    assert "if index * 20 + 20 >= pause_ms:" in source
