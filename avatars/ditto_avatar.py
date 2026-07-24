###############################################################################
#  Ditto talking-head adapter for LiveTalking
#
#  Bridges antgroup/ditto-talkinghead (StreamSDK online pipeline) into
#  LiveTalking's TTS + WebRTC output.
#
#  Key facts about the two sides (learned the hard way):
#   * Ditto's StreamSDK runs 6 worker threads; the LAST one (writer_worker)
#     calls self.writer(frame_rgb, fmt="rgb") for every finished frame. We
#     REPLACE self.sdk.writer with our own sink so those frames go to WebRTC
#     instead of a file. (Do NOT read writer_queue — that races the writer.)
#   * LiveTalking's WebRTC tracks each play on their OWN fixed clock: video at
#     25fps (VIDEO_PTIME=0.040), audio at 20ms. A/V sync therefore depends on
#     feeding both queues at real-time rate. So we pair each emitted Ditto
#     frame with the 2 audio chunks (2x20ms = 40ms = 1 frame) that produced it
#     and push them together from one 25fps pump thread. Frame[k] keeps its
#     own audio[k] → lip sync is preserved regardless of pipeline latency.
#   * Ditto only emits frames while fed audio. When idle we loop the source
#     frames (source_info["img_rgb_lst"]) + silence so the video isn't black.
#
#  Env vars:
#     DITTO_REPO       path to the cloned ditto-talkinghead repo
#     DITTO_CFG        cfg pkl (use the *pytorch* one on non-Ampere GPUs)
#     DITTO_DATA_ROOT  model dir (ditto_pytorch)
#     DITTO_PROF       log stage timings, queue sizes, and frame accounting
#     DITTO_FEED_CAP   max frames the SDK may run ahead (default 20 ≈ 0.8s). Lower
#                      → snappier interrupt + less buffered lag; too low → stutter.
#     DITTO_DEBUG      dump the first N writer + source frames and log per-frame
#                      writer-vs-source / writer-vs-previous diffs (proves whether
#                      Ditto is generating or just replaying the source video).
#     DITTO_DEBUG_DIR  where to dump (default /tmp/ditto_debug)
#     DITTO_DEBUG_N    how many writer frames to dump/compare (default 20)
#
#  Mouth/expression shaping (Ditto setup params — all opt-in; unset = baseline):
#     DITTO_EMO        emotion index (default 4; 0 = neutral)
#     DITTO_EXP        scale expression/mouth AMPLITUDE, e.g. 0.85 → smaller mouth
#                      WITHOUT blurring lip-sync (this is the knob for "small mouth
#                      AND good sync"). Builds use_d_keys with head keys kept full.
#     DITTO_SMO_K_D    temporal smoothing of driving motion — LOW (1) = sharpest
#                      lip-sync (<=1 disables it); HIGH = smaller-but-mushy mouth.
#     DITTO_SMO_K_S    smoothing of source (head/body) motion — NOT mouth-related
#     DITTO_FADE_TYPE / DITTO_OVERLAP  online-chunk blending
#
#  use_d_keys MUST include the head keys (pitch/yaw/roll/t) or the head freezes and
#  output looks like raw source playback — DITTO_EXP builds it correctly; never
#  pass a bare {"exp": ...}. If source.mp4 shows teeth/open mouth, generation
#  inherits it regardless of these knobs.
###############################################################################

import os
import sys
import time
import queue
import numpy as np
import cv2
from queue import Queue
from threading import Thread, Event

from avatars.base_avatar import BaseAvatar
from registry import register
from utils.logger import logger

_DITTO_REPO = os.environ.get("DITTO_REPO", "/workspace/ditto-talkinghead")
if _DITTO_REPO not in sys.path:
    sys.path.insert(0, _DITTO_REPO)

# 2 audio chunks (20ms each) per 25fps video frame. The WebRTC video track is
# hardwired to 25fps in server/webrtc.py, so this ratio is fixed.
_AUDIO_CHUNKS_PER_FRAME = 2
_SILENCE = np.zeros(320, dtype=np.int16)

# hubert sliding window, copied from ditto's inference.py (chunksize=(3,5,2)):
#   prepad 3*640 zeros, window = sum(chunksize)*0.04*16k + 80 = 6480,
#   hop = 5*640 = 3200 → each run_chunk emits 5 frames (5*640 = 3200, balanced).
_CHUNKSIZE = (3, 5, 2)
_PREPAD = _CHUNKSIZE[0] * 640          # 1920
_SPLIT_LEN = int(sum(_CHUNKSIZE) * 0.04 * 16000) + 80   # 6480
_HOP = _CHUNKSIZE[1] * 640             # 3200


def _tail_frame_counts(audio_chunks, scheduled_frames, batch_frames=5):
    """Return real frame counts for each fixed-size final Ditto batch."""
    remaining = max(0, (audio_chunks + 1) // 2 - scheduled_frames)
    return [min(batch_frames, remaining - i)
            for i in range(0, remaining, batch_frames)]


def load_model():
    return {
        "cfg_pkl": os.environ.get(
            "DITTO_CFG",
            f"{_DITTO_REPO}/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"),
        "data_root": os.environ.get(
            "DITTO_DATA_ROOT",
            f"{_DITTO_REPO}/checkpoints/ditto_pytorch"),
    }


def load_avatar(avatar_id):
    """For Ditto the 'avatar' is a source portrait image or video."""
    for ext in ("mp4", "png", "jpg", "jpeg"):
        p = f"./data/avatars/{avatar_id}/source.{ext}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"ditto avatar '{avatar_id}' needs data/avatars/{avatar_id}/source.mp4 (or .png)")


def warm_up(batch_size, model, *args):
    return


def _drain_queue(q):
    """Empty a Queue without blocking (used to cut buffered speech on interrupt)."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


class _FrameSink:
    """Stands in for Ditto's VideoWriterByImageIO. writer_worker calls this per
    finished RGB frame; we hand it to the avatar instead of writing a file."""
    def __init__(self, on_frame):
        self._on_frame = on_frame

    def __call__(self, frame_rgb, fmt="rgb"):
        self._on_frame(frame_rgb)

    def close(self):
        pass


class _Prof:
    """Transparent proxy that times __call__ on a Ditto pipeline stage so we can
    see which stage caps throughput. All other attribute access/set proxies to
    the wrapped object, so worker code using .d0, .cvt_fmt, .seq_frames still works."""
    def __init__(self, name, obj):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_n", 0)
        object.__setattr__(self, "_t", 0.0)
        object.__setattr__(self, "_sync", None)
        if os.environ.get("DITTO_PROF_SYNC", "1") != "0":
            try:
                import torch
                if torch.cuda.is_available():
                    object.__setattr__(self, "_sync", torch.cuda.synchronize)
            except Exception:
                pass

    def __call__(self, *a, **k):
        if self._sync:
            self._sync()
        s = time.perf_counter()
        r = self._obj(*a, **k)
        if self._sync:
            self._sync()
        object.__setattr__(self, "_t", self._t + (time.perf_counter() - s))
        object.__setattr__(self, "_n", self._n + 1)
        if self._n % 10 == 0:
            ms = self._t / self._n * 1000.0
            logger.info(f"[prof] {self._name}: {ms:.1f} ms/call → {1000.0/ms:.1f} fps cap ({self._n})")
        return r

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_obj"), k)

    def __setattr__(self, k, v):
        setattr(object.__getattribute__(self, "_obj"), k, v)


@register("avatar", "ditto")
class DittoReal(BaseAvatar):
    def __init__(self, opt, model, avatar):
        super().__init__(opt)               # wires self.tts and self.output
        self.cfg = model
        self.source_path = avatar

        self._t_build = time.perf_counter()   # [timing] session build start (= right after /offer)
        from stream_pipeline_online import StreamSDK
        _t = time.perf_counter()
        self.sdk = StreamSDK(self.cfg["cfg_pkl"], self.cfg["data_root"])
        logger.info("[ditto-timing] StreamSDK engine load: %.2fs", time.perf_counter() - _t)

        # Sliding-window buffer for hubert (see constants above). Starts with the
        # prepad so the first window's valid region lines up, exactly as inference.py.
        self._feat_buf = np.full(_PREPAD, 0.0, dtype=np.float32)
        self._feat_pos = 0

        self._ditto_frames: "Queue" = Queue()  # BGR frames out of Ditto
        self._audio_out: "Queue" = Queue()      # (float32[320], userdata) in feed order
        self._frame_keep: "Queue" = Queue()     # real audio frame=True, padded tail=False
        self._prof = bool(os.environ.get("DITTO_PROF"))
        self._prof_t0 = self._prof_last = time.perf_counter()
        self._prof_audio_chunks = 0
        self._prof_audio_samples = 0
        self._prof_run_chunks = 0
        self._prof_expected_frames = 0
        self._prof_frames_out = 0
        self._prof_frames_used = 0
        self._prof_holds = 0
        self._prof_idle = 0
        self._prof_frames_drop = 0
        self._drop_ditto_frames = 0

        # Backpressure: cap how many frames the SDK may run ahead of playback.
        # TTS delivers a whole answer's audio far faster than real time; without a
        # cap the SDK queues thousands of frames, so an interrupt has to grind the
        # old answer out before the next one starts (~10s). 20 frames ≈ 0.8s lead.
        self._feed_cap = int(os.environ.get("DITTO_FEED_CAP", "20"))
        self._feed_epoch = 0   # bumped on flush_talk to abort in-flight feeding
        self._muted = False    # set on flush; drop audio until the next utterance ('start')
        self._utt_t0 = 0.0             # [timing] utterance audio-in time
        self._utt_gen_pending = False  # log audio-in → first frame GENERATED
        self._utt_show_pending = False # log audio-in → first frame SHOWN (speak start)
        self._tts_start_seq = 0
        self._avatar_start_seq = 0
        self._utt_active = False
        self._utt_audio_chunks = 0
        self._utt_frames_scheduled = 0

        # ── diagnostics (DITTO_DEBUG) — prove writer frames ≠ source frames ──
        self._dbg = bool(os.environ.get("DITTO_DEBUG"))
        self._dbg_dir = os.environ.get("DITTO_DEBUG_DIR", "/tmp/ditto_debug")
        self._dbg_n = int(os.environ.get("DITTO_DEBUG_N", "20"))
        self._dbg_writer_saved = 0
        self._dbg_prev_writer = None
        self._dbg_src_bgr = []
        self._dbg_src_small = []

    def _sdk_queue_sizes(self):
        parts = []
        for name, obj in vars(self.sdk).items():
            if not hasattr(obj, "qsize"):
                continue
            try:
                parts.append(f"{name}={obj.qsize()}")
            except Exception:
                pass
        return " ".join(parts)

    def _prof_log(self, force=False):
        if not self._prof:
            return
        now = time.perf_counter()
        if not force and now - self._prof_last < 5.0:
            return
        self._prof_last = now
        elapsed = max(now - self._prof_t0, 0.001)
        audio_s = self._prof_audio_samples / 16000.0
        logger.info(
            "[ditto-prof] %.1fs audio=%.2fs chunks=%d run_chunk=%d expected=%d "
            "out=%d used=%d drop=%d hold=%d idle=%d out_fps=%.1f local_q ditto=%d audio=%d %s",
            elapsed, audio_s, self._prof_audio_chunks, self._prof_run_chunks,
            self._prof_expected_frames, self._prof_frames_out,
            self._prof_frames_used, self._prof_frames_drop, self._prof_holds,
            self._prof_idle, self._prof_frames_out / elapsed,
            self._ditto_frames.qsize(), self._audio_out.qsize(),
            self._sdk_queue_sizes())

    def _run_chunk(self, audio, chunksize, keep_frames=None):
        self._prof_run_chunks += 1
        self._prof_expected_frames += chunksize[1]
        if self._utt_active:
            self._utt_frames_scheduled += chunksize[1]
            keep_frames = chunksize[1] if keep_frames is None else keep_frames
            for i in range(chunksize[1]):
                self._frame_keep.put(i < keep_frames)
        self.sdk.run_chunk(audio, chunksize)
        self._prof_log()

    # TTS pushes 20ms float32 chunks here (override base, which routes to asr).
    def put_audio_frame(self, audio_chunk, datainfo: dict = {}):
        # After an interrupt we stay muted until the NEXT utterance begins, so the
        # tail of the sentence that was mid-synthesis when Stop/mic was pressed is
        # dropped instead of finishing. Each new utterance's first chunk carries
        # status='start' → unmute.
        epoch = self._feed_epoch
        if datainfo.get('status') == 'start' and not self._utt_active:
            self._muted = False
            self._utt_active = True
            self._utt_audio_chunks = 0
            self._utt_frames_scheduled = 0
            self._tts_start_seq += 1
            self._utt_t0 = time.perf_counter()   # [timing] this utterance's audio arrived
            self._utt_gen_pending = True
            self._utt_show_pending = True
        if self._muted:
            return
        # Backpressure: block the TTS feed while the SDK is >_feed_cap frames ahead,
        # so the SDK backlog (and thus interrupt latency) stays bounded. Bails if a
        # flush bumps the epoch or we're shutting down.
        while (self._prof_expected_frames - self._prof_frames_out
               - self._prof_frames_drop) >= self._feed_cap:
            if self._feed_epoch != epoch or (getattr(self, 'quit_event', None) is not None
                                             and self.quit_event.is_set()):
                return
            time.sleep(0.008)
        if self._feed_epoch != epoch:
            return
        a = np.asarray(audio_chunk, dtype=np.float32)
        if self._utt_active:
            self._utt_audio_chunks += 1
        self._prof_audio_chunks += 1
        self._prof_audio_samples += len(a)
        # queued for the speaker, in the same order it drives the mouth
        self._audio_out.put((a, datainfo))
        # accumulate and drive Ditto's mouth with a sliding 6480-sample window
        self._feat_buf = np.concatenate([self._feat_buf, a])
        while self._feat_pos + _SPLIT_LEN <= len(self._feat_buf):
            self._run_chunk(self._feat_buf[self._feat_pos:self._feat_pos + _SPLIT_LEN], _CHUNKSIZE)
            self._feat_pos += _HOP
        # drop consumed history; nothing before the next window start is needed
        if self._feat_pos:
            self._feat_buf = self._feat_buf[self._feat_pos:]
            self._feat_pos = 0
        # end of an utterance: pad-and-flush the tail so all speech gets frames,
        # then reset — otherwise leftover audio drifts into the next utterance.
        if datainfo.get('status') == 'end':
            self._flush_tail()
            self._utt_active = False

    def _flush_tail(self):
        pos = self._feat_pos
        for keep_frames in _tail_frame_counts(
                self._utt_audio_chunks, self._utt_frames_scheduled, _CHUNKSIZE[1]):
            window = self._feat_buf[pos:pos + _SPLIT_LEN]
            if len(window) < _SPLIT_LEN:
                window = np.pad(window, (0, _SPLIT_LEN - len(window)))
            self._run_chunk(window, _CHUNKSIZE, keep_frames=keep_frames)
            pos += _HOP
        self._feat_buf = np.full(_PREPAD, 0.0, dtype=np.float32)
        self._feat_pos = 0
        self._prof_log(force=True)

    def flush_talk(self):
        """Stop talking NOW (the Stop/interrupt button). The base only stops NEW
        TTS; the real speech lives in our buffers + the SDK's backlog, so we also:
          - reset the sliding-window buffer (no new chunks from buffered audio),
          - drop the frames the SDK still owes for audio already fed to run_chunk,
          - empty the generated-frame and audio queues the pump is draining.
        Without this, /interrupt_talk leaves the buffered answer playing to the end."""
        self._feed_epoch += 1                      # abort any backpressured put_audio_frame
        self._muted = True                         # drop the interrupted utterance's tail until next 'start'
        self.speaking = False
        self._tts_start_seq = 0
        self._avatar_start_seq = 0
        self._utt_active = False
        self._utt_audio_chunks = 0
        self._utt_frames_scheduled = 0
        super().flush_talk()                       # stop TTS feeding new text
        self._feat_buf = np.full(_PREPAD, 0.0, dtype=np.float32)
        self._feat_pos = 0
        pending = self._prof_expected_frames - self._prof_frames_out - self._prof_frames_drop
        if pending > 0:
            self._drop_ditto_frames += pending     # swallow the SDK's in-flight frames
        _drain_queue(self._audio_out)
        _drain_queue(self._ditto_frames)
        _drain_queue(self._frame_keep)
        logger.info("ditto flush_talk: cleared buffered speech, swallowing %d in-flight frames",
                    max(0, pending))

    def _on_frame(self, frame_rgb):
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3:
            return
        if self._drop_ditto_frames:
            self._drop_ditto_frames -= 1
            self._prof_frames_drop += 1
            self._prof_log()
            return
        try:
            keep_frame = self._frame_keep.get_nowait()
        except queue.Empty:
            keep_frame = True
        if not keep_frame:
            self._prof_frames_drop += 1
            self._prof_log()
            return
        self._prof_frames_out += 1
        if self._utt_gen_pending:
            self._utt_gen_pending = False
            logger.info("[ditto-timing] VIDEO first frame GENERATED: %.2fs after audio in (window + diffusion)",
                        time.perf_counter() - self._utt_t0)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._dbg and self._dbg_writer_saved < self._dbg_n:
            self._dbg_dump_writer(frame_bgr)
        self._ditto_frames.put(frame_bgr)
        self._prof_log()

    # ── Diagnostics (DITTO_DEBUG=1) ─────────────────────────────────────────
    # Proves whether Ditto is GENERATING or just replaying the source video by
    # dumping the first DITTO_DEBUG_N writer + source frames and logging diffs.
    # Off unless enabled; only those first N writer frames pay any cost.
    @staticmethod
    def _dbg_small(img_bgr):
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _dbg_setup_sources(self):
        os.makedirs(self._dbg_dir, exist_ok=True)
        src_rgb = self.sdk.source_info.get("img_rgb_lst") or []
        self._dbg_src_bgr = [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR) for f in src_rgb]
        self._dbg_src_small = [self._dbg_small(f) for f in self._dbg_src_bgr]
        for i, f in enumerate(self._dbg_src_bgr[:self._dbg_n]):
            cv2.imwrite(os.path.join(self._dbg_dir, f"source_{i:04d}.jpg"), f)
        logger.info("[ditto-dbg] dumped %d/%d source frames to %s",
                    min(self._dbg_n, len(self._dbg_src_bgr)),
                    len(self._dbg_src_bgr), self._dbg_dir)

    def _dbg_dump_writer(self, writer_bgr):
        i = self._dbg_writer_saved
        self._dbg_writer_saved += 1
        cv2.imwrite(os.path.join(self._dbg_dir, f"frame_{i:04d}.jpg"), writer_bgr)
        w = self._dbg_small(writer_bgr)
        # best-matching source frame — alignment-independent. If the writer is
        # merely replaying the source, SOME source frame is near-identical (~0).
        best_i, best_d = -1, 1e9
        for si, s in enumerate(self._dbg_src_small):
            d = float(np.mean(np.abs(w - s)))
            if d < best_d:
                best_d, best_i = d, si
        # same-index diff, for reference (fuzzy: the warm-up drop offsets it)
        aligned = (float(np.mean(np.abs(w - self._dbg_src_small[i])))
                   if i < len(self._dbg_src_small) else float("nan"))

        # mouth = lower-centre band, where Ditto's generated motion shows up
        def mouth(x):
            h, ww = x.shape
            return x[int(h * 0.55):int(h * 0.95), int(ww * 0.30):int(ww * 0.70)]
        mouth_d = (float(np.mean(np.abs(mouth(w) - mouth(self._dbg_src_small[best_i]))))
                   if best_i >= 0 else float("nan"))
        if best_i >= 0:
            cv2.imwrite(os.path.join(self._dbg_dir, f"match_{i:04d}_src{best_i:04d}.jpg"),
                        self._dbg_src_bgr[best_i])
        prev_d = (float(np.mean(np.abs(w - self._dbg_prev_writer)))
                  if self._dbg_prev_writer is not None else float("nan"))
        self._dbg_prev_writer = w
        logger.info("[ditto-dbg] writer#%02d best_src=%d full=%.2f mouth=%.2f "
                    "aligned=%.2f prev=%.2f (0-255 mean-abs)",
                    i, best_i, best_d, mouth_d, aligned, prev_d)
        if i == self._dbg_n - 1:
            logger.info("[ditto-dbg] DONE → %s. Read as: full≈0 every frame = writer "
                        "is replaying source (no generation). prev≈0 = static frames. "
                        "Real generation = small full/aligned but clear mouth AND "
                        "nonzero prev.", self._dbg_dir)

    def _load_idle_bgr(self):
        idle_path = os.path.join(os.path.dirname(self.source_path), "idle.mp4")
        if os.path.exists(idle_path):
            cap = cv2.VideoCapture(idle_path)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            if frames:
                logger.info("ditto idle video loaded: %s frames=%d", idle_path, len(frames))
                return frames
        return [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR)
                for f in self.sdk.source_info["img_rgb_lst"]]

    def _pump(self, quit_event: Event):
        # idle:   cycle source frames (smooth animation before/after speech)
        # speech: show Ditto frames; hold last when queue briefly empty (no flicker)
        # Real audio advances only with a generated frame. During a writer gap,
        # emit silence and hold the frame so audio cannot run ahead of the mouth.
        ii = 0
        current_frame = self._idle_bgr[0]
        last_ditto_t = 0.0
        in_speech = False
        _HOLD = float(os.environ.get("DITTO_HOLD", "0.04"))
        _START_BUFFER = int(os.environ.get("DITTO_START_BUFFER", "5"))
        # Positive offset makes the video lead the audio. A single video frame
        # is 40ms, while audio packets are 20ms.
        _AUDIO_DELAY_CHUNKS = max(0, int(round(float(os.environ.get("DITTO_AV_OFFSET_MS", "0")) / 20.0)))
        audio_delay_left = 0
        final_audio_seen = False
        pump_epoch = self._feed_epoch
        dbg_pump_saved = 0

        target = time.perf_counter()
        while not quit_event.is_set():
            now = time.perf_counter()
            got_ditto = False
            if pump_epoch != self._feed_epoch:
                pump_epoch = self._feed_epoch
                in_speech = False
                self.speaking = False
                audio_delay_left = 0
                final_audio_seen = False
            generation_pending = (
                self._prof_expected_frames >
                self._prof_frames_out + self._prof_frames_drop
            )
            try:
                if not in_speech and self._ditto_frames.qsize() < _START_BUFFER:
                    raise queue.Empty
                current_frame = self._ditto_frames.get_nowait()
                got_ditto = True
                if not in_speech:
                    audio_delay_left = _AUDIO_DELAY_CHUNKS
                    final_audio_seen = False
                in_speech = True
                self.speaking = True
                last_ditto_t = now
                self._prof_frames_used += 1
                if self._utt_show_pending:
                    self._utt_show_pending = False
                    self._avatar_start_seq += 1
                    logger.info("[ditto-timing] SPEAK START (audio+video out to WebRTC): %.2fs after audio in",
                                time.perf_counter() - self._utt_t0)
                # confirm what's shown DURING speech is a writer frame (not idle):
                # these should visually equal frame_*.jpg from _dbg_dump_writer.
                if self._dbg and dbg_pump_saved < self._dbg_n:
                    cv2.imwrite(os.path.join(self._dbg_dir,
                                f"pump_ditto_{dbg_pump_saved:04d}.jpg"), current_frame)
                    dbg_pump_saved += 1
            except queue.Empty:
                if (in_speech and final_audio_seen and self._audio_out.empty()
                        and not generation_pending and (now - last_ditto_t) > _HOLD):
                    in_speech = False  # speech done, resume idle
                    self.speaking = False
                    audio_delay_left = 0
                    logger.info("ditto pump: speech drained -> idle")
                if not in_speech:
                    self.speaking = False
                    idle_frame = self._idle_bgr[ii % len(self._idle_bgr)]
                    ii += 1
                    current_frame = idle_frame
                    self._prof_idle += 1
                else:
                    self._prof_holds += 1
                # else: mid-speech gap — hold last Ditto frame (no flicker)

            self.output.push_video_frame(current_frame)
            self.record_video_data(current_frame)

            for _ in range(_AUDIO_CHUNKS_PER_FRAME):
                drain_offset_tail = (
                    in_speech and _AUDIO_DELAY_CHUNKS and not got_ditto
                    and not generation_pending
                )
                if in_speech and (got_ditto or drain_offset_tail):
                    if audio_delay_left:
                        audio_delay_left -= 1
                        pcm, ud = _SILENCE, {}
                    else:
                        try:
                            a, ud = self._audio_out.get_nowait()
                            pcm = (a * 32767).astype(np.int16)
                        except queue.Empty:
                            pcm, ud = _SILENCE, {}
                    if ud.get("status") == "end":
                        final_audio_seen = True
                else:
                    pcm, ud = _SILENCE, {}
                self.output.push_audio_frame(pcm, ud)
                self.record_audio_data(pcm)

            target += 0.04
            dt = target - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                target = time.perf_counter()
            self._prof_log()
        logger.info('ditto pump stop')
        self.speaking = False
        self._prof_log(force=True)

    def render(self, quit_event):
        self.quit_event = quit_event
        self.init_customindex()

        # Register source; output_path is a dummy — we replace the writer below.
        # Two speed knobs (env-tunable) — the pytorch backend can't hit 25fps at
        # full res/steps, and below-real-time output makes the mouth stutter:
        #   DITTO_STEPS     LMDM diffusion denoise steps (default 50). Biggest
        #                   lever; 15 is ~3x faster with little visible loss.
        #   DITTO_MAX_SIZE  longest-edge the pipeline processes/outputs at
        #                   (default 1920). 640 is plenty for a talking head.
        # Base call = the working "大牙" baseline: Ditto's DEFAULT motion keys.
        # Do NOT add use_d_keys here — it restricts the applied keys and flattens
        # the mouth so the output looks like raw source playback (reverted).
        setup_kwargs = dict(
            sampling_timesteps=int(os.environ.get("DITTO_STEPS", "8")),
            max_size=int(os.environ.get("DITTO_MAX_SIZE", "640")),
            emo=int(os.environ.get("DITTO_EMO", "4")),
            online_mode=os.environ.get("DITTO_ONLINE", "0") == "1",
        )
        # Official mouth/expression-shaping knobs — the correct way to tame the
        # mouth (never use_d_keys). Forwarded ONLY when the env var is set, so the
        # default call stays identical to the baseline. If a name doesn't match the
        # installed Ditto build, setup() errors only when you opt into that var.
        #   DITTO_FADE_TYPE  crossfade style between online chunks (e.g. d0)
        #   DITTO_OVERLAP    online chunk overlap (overlap_v2, frames)
        #   DITTO_SMO_K_D    smoothing kernel over driving motion
        #   DITTO_SMO_K_S    smoothing kernel over source motion
        if os.environ.get("DITTO_FADE_TYPE"):
            setup_kwargs["fade_type"] = os.environ["DITTO_FADE_TYPE"]
        if os.environ.get("DITTO_OVERLAP"):
            setup_kwargs["overlap_v2"] = int(os.environ["DITTO_OVERLAP"])
        if os.environ.get("DITTO_SMO_K_D"):
            setup_kwargs["smo_k_d"] = int(os.environ["DITTO_SMO_K_D"])
        if os.environ.get("DITTO_SMO_K_S"):
            setup_kwargs["smo_k_s"] = int(os.environ["DITTO_SMO_K_S"])
        # DITTO_EXP: scale ONLY the expression/mouth amplitude → smaller mouth
        # WITHOUT blurring lip-sync (unlike smo_k_d, which smears the shapes).
        # Keep the head keys (pitch/yaw/roll/t) at full: the old bug passed a bare
        # {"exp": x}, dropping them, so the head froze and it looked like raw
        # source playback. motion_stitch applies (v - d0[k]) * use_d_keys[k] per key.
        if os.environ.get("DITTO_EXP"):
            _e = float(os.environ["DITTO_EXP"])
            setup_kwargs["use_d_keys"] = {"exp": _e, "pitch": 1.0, "yaw": 1.0, "roll": 1.0, "t": 1.0}
        logger.info("ditto setup kwargs: %s", setup_kwargs)
        _t = time.perf_counter()
        self.sdk.setup(self.source_path, f"/tmp/ditto_{self.opt.sessionid}.mp4",
                       **setup_kwargs)
        logger.info("[ditto-timing] sdk.setup (source processing): %.2fs", time.perf_counter() - _t)
        # Hijack Ditto's file writer → frames flow to WebRTC (no queue race).
        self.sdk.writer = _FrameSink(self._on_frame)
        if self._dbg:
            self._dbg_setup_sources()
        # DITTO_PROF=1 → time each per-frame stage to find the throughput cap.
        if os.environ.get("DITTO_PROF"):
            for _s in ("audio2motion", "motion_stitch", "warp_f3d", "decode_f3d", "putback"):
                setattr(self.sdk, _s, _Prof(_s, getattr(self.sdk, _s)))
            logger.info("ditto profiling ON")
        # Idle frames = the full source frames Ditto composites onto (RGB→BGR).
        self._idle_bgr = self._load_idle_bgr()

        # Trigger PyTorch JIT compilation now (while idle) so the first real
        # utterance doesn't hit the cold-start penalty (~2fps for first 5-10s).
        self._drop_ditto_frames += _CHUNKSIZE[1]
        self._run_chunk(np.zeros(_SPLIT_LEN, dtype=np.float32), _CHUNKSIZE)
        logger.info("ditto JIT warm-up chunk queued")

        self.tts.render(quit_event)          # TTS → put_audio_frame → run_chunk
        self.output.start()

        pump_quit = Event()
        pump = Thread(target=self._pump, args=(pump_quit,))
        pump.start()

        logger.info("[ditto-timing] total build→ready: %.2fs (engine load + setup + warmup queued)",
                    time.perf_counter() - self._t_build)
        logger.info('ditto render start')
        while not quit_event.is_set():
            time.sleep(0.1)
        logger.info('ditto render stop')

        pump_quit.set()
        pump.join()
        try:
            self.sdk.close()
        except Exception:
            logger.exception("ditto sdk close error")
        self.output.stop()

# ponytail: _ditto_frames / _audio_out are unbounded. If TTS outruns real-time
# on a long utterance they grow (latency creeps up, sync holds). Cap with a
# bounded Queue + drop-oldest only if that actually bites in a demo.
