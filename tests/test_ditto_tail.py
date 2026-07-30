import ast
import os
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

    assert "self._frame_keep.put((keep, frame_seq))" in source
    assert "_take_audio_pair(self._audio_out)" in source
    assert "if not keep_frame:" in source
    assert "if in_speech:" in source
    assert "if in_speech and got_ditto:" not in source
    assert "self._utt_active or not self._audio_out.empty()" in source
    assert "DITTO_IDLE_DELAY" not in source
    assert 'DITTO_START_BUFFER", "8"' in source
    assert "DITTO_IDLE_FADE_MS" not in source
    assert "cv2.addWeighted" not in source
    assert 'DITTO_AV_OFFSET_MS", "100"' in source
    assert "_take_due_frame" not in source
    assert "final_audio_seen" not in source
    assert "if not self._speech_pending() and not delayed_audio:" in source
    assert "self._audio_cap" not in source
    assert "self._audio_out.qsize() >=" not in source
    assert "ditto stop fence" not in source
    assert "valid_clip_len = self.sdk.audio2motion.valid_clip_len" not in source
    assert "self._drop_ditto_frames += _CHUNKSIZE[1]" in source
    assert "_neutralize_source_lips" not in source
    assert '"vad_dst": self._vad_dst[source_index]' in source
    assert 'exp = info["exp"].copy()' in source
    assert "self._ctrl_frame_next = self._prof_expected_frames" in source


def test_tts_silence_tail_marks_only_its_final_frame():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    assert "for index in range((pause_ms + 19) // 20):" in source
    assert "if index * 20 + 20 >= pause_ms:" in source
    assert "ditto_vad" not in source
    assert '"_ditto_silence": True' in source
    assert 'DITTO_VAD_RMS", "0.006"' in source
    assert 'DITTO_VAD_MIN_MS", "80"' in source
    assert "len(quiet_frames) >= quiet_frames_needed" in source


def test_tts_silence_tail_preserves_final_end_marker():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ElevenLabsTTS"
    )
    method = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_send_silence_tail"
    )
    namespace = {"np": np, "os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<tts-tail>", "exec"), namespace)

    events = []

    class Parent:
        @staticmethod
        def put_audio_frame(frame, event):
            events.append((frame, event))

    tts = type("TTS", (), {"chunk": 320, "parent": Parent()})()
    namespace["_send_silence_tail"](
        tts,
        "Done.",
        {"pause_ms": 50, "final": True},
        True,
    )

    assert len(events) == 3
    assert all(event["_ditto_silence"] for _, event in events)
    assert "status" not in events[0][1]
    assert events[-1][1]["status"] == "end"
    assert events[-1][1]["text"] == "Done."


def test_idle_transition_matches_recent_motion():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_frame_small", "_closest_idle_sequence_index"}
    }
    namespace = {"np": np, "cv2": __import__("cv2")}
    exec(
        compile(
            ast.Module(
                body=[
                    functions["_frame_small"],
                    functions["_closest_idle_sequence_index"],
                ],
                type_ignores=[],
            ),
            "<idle-match>",
            "exec",
        ),
        namespace,
    )

    dark = np.zeros((16, 16, 3), dtype=np.uint8)
    light = np.full((16, 16, 3), 255, dtype=np.uint8)
    idle_small = [
        namespace["_frame_small"](dark),
        namespace["_frame_small"](light),
    ]
    history = [
        namespace["_frame_small"](dark),
        namespace["_frame_small"](light),
    ]
    assert namespace["_closest_idle_sequence_index"](history, idle_small) == 0


def test_generated_frame_takes_exactly_two_audio_packets():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_take_audio_pair"
    )
    namespace = {
        "queue": __import__("queue"),
        "_AUDIO_CHUNKS_PER_FRAME": 2,
        "_SILENCE_FLOAT": np.zeros(320, dtype=np.float32),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<audio-pair>", "exec"), namespace)

    audio = namespace["queue"].Queue()
    audio.put(("audio-0", {"status": "start"}))
    audio.put(("audio-1", {}))
    audio.put(("audio-2", {"status": "end"}))

    first = namespace["_take_audio_pair"](audio)
    second = namespace["_take_audio_pair"](audio)

    assert first == [("audio-0", {"status": "start"}), ("audio-1", {})]
    assert second[0] == ("audio-2", {"status": "end"})
    assert np.array_equal(second[1][0], namespace["_SILENCE_FLOAT"])


def test_vad_requires_a_full_silent_video_frame():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_vad_frame_values"
    )
    namespace = {"np": np, "_CHUNKSIZE": (3, 5, 2), "_PREPAD": 1920}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<vad>", "exec"), namespace)

    window = np.zeros(6480, dtype=np.float32)
    window[1920:2240] = 1.0
    assert namespace["_vad_frame_values"](window) == [1.0, 0.0, 0.0, 0.0, 0.0]

    mirror = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_source_frame_index"
    )
    exec(compile(ast.Module(body=[mirror], type_ignores=[]), "<mirror>", "exec"), namespace)
    assert [namespace["_source_frame_index"](i, 3) for i in range(8)] == [
        0, 1, 2, 2, 1, 0, 0, 1
    ]


def test_docker_patches_vad_to_use_an_isolated_neutral_target():
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")
    assert 'kwargs.get("vad_dst", x_s_info)' in dockerfile
