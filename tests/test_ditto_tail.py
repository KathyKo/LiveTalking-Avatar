import ast
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
    assert "if in_speech:" in source
    assert "if in_speech and got_ditto:" not in source
    assert "return not self._audio_out.empty()" in source
    assert 'DITTO_HOLD", "0.10"' in source
    assert 'DITTO_START_BUFFER", "8"' in source
    assert "DITTO_IDLE_FADE_MS" not in source
    assert "cv2.addWeighted" not in source
    assert 'DITTO_AV_OFFSET_MS", "60"' in source
    assert "self._audio_cap" not in source
    assert "self._audio_out.qsize() >=" not in source
    assert "ditto stop fence" not in source
    assert "valid_clip_len = self.sdk.audio2motion.valid_clip_len" in source
    assert "self._drop_ditto_frames += _CHUNKSIZE[1]" not in source


def test_tts_silence_tail_marks_only_its_final_frame():
    source = (Path(__file__).parents[1] / "tts" / "elevenlabs_tts.py").read_text(
        encoding="utf-8"
    )
    assert "for index in range((pause_ms + 19) // 20):" in source
    assert "if index * 20 + 20 >= pause_ms:" in source
    assert 'eventpoint = {"ditto_vad": 0.0}' in source


def test_vad_and_neutral_source_lips():
    source = (Path(__file__).parents[1] / "avatars" / "ditto_avatar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_vad_frame_values", "_neutralize_source_lips"}
    }
    namespace = {
        "np": np,
        "_CHUNKSIZE": (3, 5, 2),
        "_PREPAD": 1920,
        "_LIP_POINTS": (6, 12, 14, 17, 19, 20),
    }
    exec(
        compile(ast.Module(body=list(functions.values()), type_ignores=[]), "<sync>", "exec"),
        namespace,
    )

    vad = np.concatenate([
        np.ones(1920),
        np.ones(640),
        np.full(640, 0.5),
        np.zeros(640 * 3),
        np.ones(80),
    ])
    assert namespace["_vad_frame_values"](vad) == [1.0, 0.5, 0.0, 0.0, 0.0]

    first = np.arange(63, dtype=np.float32).reshape(1, 63)
    neutral = first + 100
    source_info = {
        "x_s_info_lst": [{"exp": first.copy()}, {"exp": neutral.copy()}]
    }
    assert namespace["_neutralize_source_lips"](source_info, 1) == 1
    result = source_info["x_s_info_lst"][0]["exp"].reshape(21, 3)
    expected = neutral.reshape(21, 3)
    assert np.array_equal(
        result[[6, 12, 14, 17, 19, 20]],
        expected[[6, 12, 14, 17, 19, 20]],
    )
    assert np.array_equal(result[0], first.reshape(21, 3)[0])
