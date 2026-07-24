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
    assert "if in_speech and (got_ditto or drain_offset_tail):" in source
    assert 'ud.get("status") == "end"' in source
    assert "in_speech and final_audio_seen and self._audio_out.empty()" in source
    assert "pump_epoch != self._feed_epoch" in source
    assert 'DITTO_HOLD", "0.04"' in source
    assert 'DITTO_START_BUFFER", "5"' in source
    assert "DITTO_IDLE_FADE_MS" not in source
    assert "cv2.addWeighted" not in source
    assert 'DITTO_AV_OFFSET_MS", "0"' in source
    assert "self._audio_cap" not in source
    assert "self._audio_out.qsize() >=" not in source
    assert "ditto stop fence" not in source


def test_tts_silence_tail_marks_only_its_final_frame():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    assert 'DITTO_TAIL_MS", "300"' in source
    assert "for index in range((pause_ms + 19) // 20):" in source
    assert "if index * 20 + 20 >= pause_ms:" in source
