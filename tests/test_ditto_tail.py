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
    # Audio is bound to its frame when the frame is generated, and the pump only
    # ever plays the audio carried by the frame it is showing. Draining the two
    # from separate queues slips 40ms every time either side starves.
    assert "self._ditto_frames.put((frame_bgr, _take_audio_pair(self._audio_out)))" in source
    assert "audio_delay.extend(frame_audio)" in source
    assert "self._audio_out.get_nowait()" not in source.split("def _pump")[1]
    assert "if in_speech and got_ditto:" not in source
    # Idle only once the SDK owes no frames AND nothing is left to play.
    assert "return not self._audio_out.empty() or not self._ditto_frames.empty()" in source
    assert 'DITTO_HOLD", "0.10"' in source
    assert 'DITTO_START_BUFFER", "8"' in source
    assert "DITTO_IDLE_FADE_MS" not in source
    assert "cv2.addWeighted" not in source
    assert "final queues drained; holding last frame" in source
    assert "and self._audio_out.empty() and self._ditto_frames.empty()" in source
    assert "and not self._final_pending" in source
    assert 'DITTO_FINAL_HOLD_MS", "500"' in source
    assert "final audio played; holding last frame" in source
    assert "_END_HOLD = max(_HOLD, _AUDIO_DELAY_CHUNKS * 0.02)" in source
    assert "while (np.any(a) and" in source
    assert "ditto final audio received: flushing tail" in source
    neutralize = source.split("def _neutralize_source_lips", 1)[1].split("def _pump", 1)[0]
    assert "audio2motion.setup(" not in neutralize
    assert 'DITTO_AV_OFFSET_MS", "60"' in source
    assert "self._audio_cap" not in source
    assert "self._audio_out.qsize() >=" not in source
    assert "ditto stop fence" not in source


def test_alignment_flush_lands_on_sdk_batch_boundary():
    """An utterance must never strand half a 10-frame batch inside the SDK."""
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_alignment_flush_chunks"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<align>", "exec"), namespace)
    flush = namespace["_alignment_flush_chunks"]

    assert flush(2, 10, 5) == 0      # already on a boundary → no padding
    assert flush(3, 10, 5) == 1      # 15 frames fed, 5 stranded → one chunk
    for run_chunks in range(200):
        pad = flush(run_chunks, 10, 5)
        assert pad <= 1, "padding must never exceed one batch"
        assert (run_chunks + pad) * 5 % 10 == 0, "utterance still strands a half batch"

    # padding frames must be discarded, never paired with real audio
    assert "keep_frames=0)" in source


def test_tts_silence_tail_marks_only_its_final_frame():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    assert "for index in range((pause_ms + 19) // 20):" in source
    assert "if index * 20 + 20 >= pause_ms:" in source
    assert 'status="end" if final else "segment_end"' in source
    assert "final=final" in source
