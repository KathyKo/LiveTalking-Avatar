import ast
import time
from queue import Queue
from threading import Event, Thread
from pathlib import Path

import numpy as np


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
    assert 'DITTO_START_BUFFER", "6"' in source
    assert "DITTO_IDLE_FADE_MS" not in source
    assert "idle_blend" in source
    assert "final queues drained; holding" in source
    assert "and self._audio_out.empty() and self._ditto_frames.empty()" in source
    assert "and not self._final_pending" in source
    assert 'DITTO_FINAL_HOLD_MS", "370"' in source
    assert "final audio played; holding" in source
    assert "_END_HOLD = max(_HOLD, _AUDIO_DELAY_CHUNKS * 0.02)" in source
    assert "while (np.any(a) and" in source
    assert "ditto final audio received: flushing tail" in source
    neutralize = source.split("def _neutralize_source_lips", 1)[1].split("def _pump", 1)[0]
    assert "audio2motion.setup(" not in neutralize
    assert 'DITTO_AV_OFFSET_MS", "260"' in source
    assert "self._audio_cap" not in source
    assert "self._audio_out.qsize() >=" not in source
    assert "ditto stop fence" not in source
    assert "SDK tail stalled; draining final audio" in source


def test_idle_transition_matches_pose_before_blending():
    from avatars.ditto_avatar import (
        _blend_to_idle,
        _closest_idle_index,
        _frame_thumb,
        _resize_idle_frame,
    )

    final = np.full((8, 8, 3), 190, dtype=np.uint8)
    idle = [np.zeros_like(final), np.full_like(final, 200)]
    assert _closest_idle_index(final, [_frame_thumb(frame) for frame in idle]) == 1
    transition = _blend_to_idle(final, [idle[1]] * 4)
    assert len(transition) == 4
    assert all(np.mean(transition[i]) < np.mean(transition[i + 1])
               for i in range(3))
    assert _blend_to_idle(final, [np.zeros((4, 4, 3), dtype=np.uint8)]) == []

    recorded_idle = np.zeros((16, 24, 3), dtype=np.uint8)
    fitted_idle = _resize_idle_frame(recorded_idle, final.shape)
    assert fitted_idle.shape == final.shape
    assert len(_blend_to_idle(final, [fitted_idle] * 4)) == 4


def test_lip_response_sharpens_only_lip_motion():
    from avatars.ditto_avatar import _LIP_KEYPOINTS, _sharpen_lip_sequence

    sequence = np.zeros((1, 3, 265), dtype=np.float32)
    expression = sequence[..., -63:].reshape(1, 3, 21, 3)
    expression[0, 1, list(_LIP_KEYPOINTS), :] = 1.0
    expression[0, 2, list(_LIP_KEYPOINTS), :] = 1.5
    expression[0, :, 0, :] = np.array([0.0, 2.0, 4.0])[:, None]

    sharpened, previous = _sharpen_lip_sequence(sequence, 1.15)
    sharpened_exp = sharpened[..., -63:].reshape(1, 3, 21, 3)

    np.testing.assert_allclose(sharpened_exp[0, 0, list(_LIP_KEYPOINTS), :], 0.0)
    np.testing.assert_allclose(sharpened_exp[0, 1, list(_LIP_KEYPOINTS), :], 1.15)
    np.testing.assert_allclose(sharpened_exp[0, 2, list(_LIP_KEYPOINTS), :], 1.575)
    np.testing.assert_allclose(sharpened_exp[0, :, 0, :], expression[0, :, 0, :])
    np.testing.assert_allclose(previous, 1.5)


def test_semantic_pause_closes_only_lips_over_three_frames():
    from avatars.ditto_avatar import _LIP_KEYPOINTS, _SemanticPauseMotion

    class Stitch:
        def __call__(self, source, driving, **_kwargs):
            return source, driving

    pauses = iter([True, True, True, False])
    wrapper = _SemanticPauseMotion(Stitch(), lambda: next(pauses), 3)

    source = np.zeros((1, 21, 3), dtype=np.float32)
    driving = np.ones((1, 21, 3), dtype=np.float32)
    outputs = [wrapper(source, driving)[1] for _ in range(4)]
    lips = [out[:, list(_LIP_KEYPOINTS), :] for out in outputs]

    np.testing.assert_allclose(lips[0], 2.0 / 3.0)
    np.testing.assert_allclose(lips[1], 1.0 / 3.0)
    np.testing.assert_allclose(lips[2], 0.0)
    np.testing.assert_allclose(lips[3], 1.0)
    np.testing.assert_allclose(outputs[2][:, 0, :], 1.0)


def test_semantic_pause_requires_a_full_40ms_silence_frame():
    from avatars.ditto_avatar import DittoReal

    avatar = object.__new__(DittoReal)
    avatar._pause_frames = Queue()
    avatar._pause_packets = []

    avatar._queue_pause_packet({})
    avatar._queue_pause_packet({"semantic_pause": True})
    assert avatar._next_pause_frame() is False

    avatar._queue_pause_packet({"semantic_pause": True})
    avatar._queue_pause_packet({"semantic_pause": True})
    assert avatar._next_pause_frame() is True


def test_pump_drains_stranded_final_audio_and_returns_idle(monkeypatch):
    from avatars.ditto_avatar import DittoReal

    monkeypatch.setenv("DITTO_START_BUFFER", "1")
    monkeypatch.setenv("DITTO_AV_OFFSET_MS", "0")
    monkeypatch.setenv("DITTO_FINAL_HOLD_MS", "370")

    events = []

    class Output:
        def push_video_frame(self, frame):
            events.append((time.perf_counter(), "video", int(frame[0, 0, 0])))

        def push_audio_frame(self, _pcm, data):
            events.append((time.perf_counter(), "audio", data))

    avatar = object.__new__(DittoReal)
    avatar._idle_bgr = [np.zeros((2, 2, 3), dtype=np.uint8)]
    avatar._ditto_frames = Queue()
    avatar._audio_out = Queue()
    avatar._ditto_frames.put((np.full((2, 2, 3), 255, dtype=np.uint8),
                              [(None, {}), (None, {})]))
    avatar._audio_out.put((np.ones(320, dtype=np.float32), {}))
    avatar._audio_out.put((np.ones(320, dtype=np.float32),
                           {"status": "end", "final": True}))
    avatar._final_pending = True
    avatar._utt_active = False
    avatar._last_ditto_frame_at = time.perf_counter() - 1.0
    avatar._tail_audio_fallback = False
    avatar._utt_show_pending = False
    avatar._dbg = False
    avatar._sync_csv = None
    avatar._prof_frames_used = avatar._prof_holds = avatar._prof_idle = 0
    avatar.speaking = False
    avatar.output = Output()
    avatar.record_video_data = lambda _frame: None
    avatar.record_audio_data = lambda _pcm: None
    avatar._prof_log = lambda force=False: None

    quit_event = Event()
    thread = Thread(target=avatar._pump, args=(quit_event,))
    thread.start()
    deadline = time.perf_counter() + 1.5
    while time.perf_counter() < deadline:
        final = next((t for t, kind, data in events
                      if kind == "audio" and data.get("final")), None)
        if final is not None and any(t > final and kind == "video" and data == 0
                                     for t, kind, data in events):
            break
        time.sleep(0.01)
    quit_event.set()
    thread.join(timeout=1.0)

    final = next(t for t, kind, data in events
                 if kind == "audio" and data.get("final"))
    idle = next(t for t, kind, data in events
                if t > final and kind == "video" and data == 0)
    assert idle - final >= 0.35
    assert idle - final < 0.70
    assert any(t > final and kind == "video" and 0 < data < 255
               for t, kind, data in events)
    assert not avatar._final_pending


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


def test_flush_does_not_double_reserve_warmup_frames():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reserve_drop_frames"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<drop>", "exec"), namespace)
    reserve = namespace["_reserve_drop_frames"]

    assert reserve(5, 5) == 5   # warm-up was already reserved
    assert reserve(5, 20) == 20
    assert reserve(0, 20) == 20


def test_tts_silence_tail_marks_only_its_final_frame():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    assert 'DITTO_TAIL_MS", "500"' in source
    assert "for index in range((pause_ms + 19) // 20):" in source
    assert "if index * 20 + 20 >= pause_ms:" in source
    assert 'status="end" if final else "segment_end"' in source
    assert "final=final" in source
    assert 'eventpoint["semantic_pause"] = True' in source


def _load_segment_gain():
    """_segment_gain without importing the module (needs elevenlabs + API key)."""
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = [
        node for node in tree.body
        if (isinstance(node, ast.Assign)
            and any(getattr(t, "id", "").startswith(("_TARGET", "_MAX", "_CEILING"))
                    for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name == "_segment_gain")
    ]
    namespace = {"np": np, "os": __import__("os")}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "<gain>", "exec"), namespace)
    return namespace["_segment_gain"], namespace["_TARGET_RMS"]


def test_segment_gain_matches_loudness_across_segments():
    """Hot and quiet segments must land at the same level, without clipping."""
    gain, target = _load_segment_gain()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(3200).astype(np.float32)
    noise /= np.max(np.abs(noise))

    def leveled(rms):
        scaled = noise * (rms / float(np.sqrt(np.mean(np.square(noise)))))
        frames = [scaled[i:i + 320] for i in range(0, 3200, 320)]
        g = gain(frames)
        out = np.concatenate([np.clip(f * g, -1.0, 1.0) for f in frames])
        return float(np.sqrt(np.mean(np.square(out)))), float(np.max(np.abs(out)))

    hot_rms, hot_peak = leveled(0.30)     # "Certainly!"
    quiet_rms, quiet_peak = leveled(0.02)  # a long flat sentence
    assert abs(hot_rms - quiet_rms) < 0.2 * target, (hot_rms, quiet_rms)
    assert hot_peak <= 1.0 and quiet_peak <= 1.0
    # Silence must pass through untouched, not be amplified into noise.
    assert gain([np.zeros(320, dtype=np.float32)]) == 1.0
    assert gain([]) == 1.0
