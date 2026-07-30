import ast
from collections import deque
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
    assert "current_frame, audio_pair = self._ditto_frames.get_nowait()" in source
    assert "self._ditto_frames.put((frame_bgr, audio_pair))" in source
    assert "_audio_tick(shifted_audio, new_audio, drain_offset_tail)" in source
    assert "or self._video_pending()" in source
    assert 'DITTO_HOLD", "0.10"' in source
    assert 'DITTO_START_BUFFER", "8"' in source
    assert "DITTO_IDLE_FADE_MS" not in source
    assert "cv2.addWeighted" not in source
    assert 'DITTO_AV_OFFSET_MS", "60"' in source
    assert "self._audio_cap" not in source
    assert "self._audio_out.qsize() >=" not in source
    assert "ditto stop fence" not in source


def test_tts_silence_tail_marks_only_its_final_frame():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    assert "for index in range((pause_ms + 19) // 20):" in source
    assert "if index * 20 + 20 >= pause_ms:" in source


def test_paired_audio_waits_on_gaps_and_drains_offset_tail():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_audio_tick"
    )
    namespace = {"_AUDIO_CHUNKS_PER_FRAME": 2}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<sync>", "exec"), namespace)
    tick = namespace["_audio_tick"]

    pending = deque(["delay-1", "delay-2", "delay-3"])
    assert tick(pending, ["audio-1", "audio-2"]) == ["delay-1", "delay-2"]
    assert tick(pending) == [None, None]
    assert list(pending) == ["delay-3", "audio-1", "audio-2"]
    assert tick(pending, drain_tail=True) == ["delay-3", "audio-1"]
    assert tick(pending, drain_tail=True) == ["audio-2", None]
    assert not pending
