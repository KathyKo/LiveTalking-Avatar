"""Render idle.mp4 through Ditto itself, so switching to idle has no seam.

The seam you see when speech ends is not a timing bug. self._idle_bgr comes from
a hand-recorded idle.mp4, while speech frames come out of Ditto's warp network +
decoder at max_size. Those are two different renderings of the same person, so
cutting or crossfading between them shows a change in crop, sharpness and colour
no amount of hold/blend tuning can hide.

Feeding silence through the same SDK, with the same setup kwargs, produces idle
frames that ARE decoder output — pixel-consistent with speech by construction.

On the pod, in a JupyterLab terminal:

    cd /opt/livetalking
    source docker/ditto-env.sh     # REQUIRED — see below
    mv data/avatars/ditto_woman/idle.mp4 data/avatars/ditto_woman/idle.mp4.bak
    python scripts/ditto_make_idle.py --avatar ditto_woman

The source line is not optional. start.sh exports DITTO_* inside the server's
own process, so a fresh shell does not see them and this script would silently
fall back to different defaults (EMO=4, ONLINE=0 instead of the server's 0 and
1) — producing an idle clip that does not match, with no error. It prints the
kwargs it used; check them against the server's "ditto setup kwargs:" log line.

Re-run it whenever DITTO_MAX_SIZE / DITTO_STEPS / DITTO_EXP / DITTO_EMO change.
data/ is a symlink to the network volume, so the result survives pod restarts.
"""

import os
import sys
import time
import argparse
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avatars.ditto_avatar import setup_kwargs_from_env   # noqa: E402

CHUNKSIZE = (3, 5, 2)
SPLIT_LEN = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80
FPS = 25


class Collector:
    """Ditto's writer_worker is a single thread, so append order is frame order."""

    def __init__(self):
        self.frames = []
        self.last = None
        self.lock = threading.Lock()

    def __call__(self, frame_rgb, fmt="rgb"):
        with self.lock:
            self.frames.append(np.asarray(frame_rgb))
            self.last = time.perf_counter()

    def close(self):
        pass


def ping_pong(frames):
    """Forward then backward, so looping the clip never hard-cuts.

    The pump plays _idle_bgr[ii % len], so frame -1 butts straight against
    frame 0. Mirroring makes the two ends identical instead.
    ponytail: doubles the frame count. Fine at a few seconds; if idle ever needs
    to be long, cross-dissolve the ends instead of mirroring.
    """
    return frames + frames[-2:0:-1] if len(frames) > 2 else frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--avatar", default=os.environ.get("AVATAR_ID", "ditto_woman"))
    ap.add_argument("--data-root", default=os.environ.get("DITTO_AVATAR_DATA", "data/avatars"))
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="silence to render; ping-pong doubles the loop length")
    ap.add_argument("--out", default=None, help="default: <avatar dir>/idle.mp4")
    ap.add_argument("--ditto-repo", default=os.environ.get("DITTO_REPO", "/opt/ditto-talkinghead"))
    ap.add_argument("--cfg", default=os.environ["DITTO_CFG"] if "DITTO_CFG" in os.environ else None)
    ap.add_argument("--sdk-data-root", default=os.environ.get("DITTO_DATA_ROOT"))
    ap.add_argument("--no-ping-pong", action="store_true")
    args = ap.parse_args()

    if not args.cfg or not args.sdk_data_root:
        sys.exit("set DITTO_CFG and DITTO_DATA_ROOT (or pass --cfg/--sdk-data-root)")

    avatar_dir = os.path.join(args.data_root, args.avatar)
    sources = [os.path.join(avatar_dir, name) for name in sorted(os.listdir(avatar_dir))
               if name.startswith("source.")]
    if not sources:
        sys.exit(f"no source.* in {avatar_dir}")
    source, out = sources[0], args.out or os.path.join(avatar_dir, "idle.mp4")
    if os.path.exists(out):
        # Never silently destroy the clip that is currently working.
        sys.exit(f"{out} exists — move it aside first (that is also your rollback)")

    sys.path.insert(0, args.ditto_repo)
    from stream_pipeline_online import StreamSDK

    kwargs = setup_kwargs_from_env()
    print(f"source={source}\nkwargs={kwargs}")
    sdk = StreamSDK(args.cfg, args.sdk_data_root)
    sdk.setup(source, "/tmp/ditto_make_idle_dummy.mp4", **kwargs)

    collector = Collector()
    sdk.writer = collector

    wanted = int(args.seconds * FPS)
    runs = (wanted + CHUNKSIZE[1] - 1) // CHUNKSIZE[1]
    silence = np.zeros(SPLIT_LEN, dtype=np.float32)
    for _ in range(runs):
        sdk.run_chunk(silence, CHUNKSIZE)

    # The SDK generates asynchronously; wait until it stops producing.
    deadline = time.perf_counter() + max(120.0, args.seconds * 20)
    while time.perf_counter() < deadline:
        with collector.lock:
            n, last = len(collector.frames), collector.last
        if n >= runs * CHUNKSIZE[1]:
            break
        if last and time.perf_counter() - last > 15:
            print(f"SDK stopped producing at {n} frames")
            break
        time.sleep(0.2)

    with collector.lock:
        frames = list(collector.frames)
    # Ditto's online mode warms up on the first batch; those frames are the ones
    # the server drops at speech start, so they do not belong in a loop either.
    frames = frames[CHUNKSIZE[1] * 2:]
    if len(frames) < FPS:
        sys.exit(f"only {len(frames)} usable frames — check cfg/data-root and GPU")
    if not args.no_ping_pong:
        frames = ping_pong(frames)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        sys.exit(f"cannot open {out} for writing")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    try:
        sdk.close()
    except Exception:
        pass
    print(f"wrote {out}: {len(frames)} frames @ {FPS}fps, {width}x{height} "
          f"({len(frames) / FPS:.1f}s loop)")


def selfcheck():
    assert ping_pong([1, 2, 3, 4]) == [1, 2, 3, 4, 3, 2]
    assert ping_pong([1, 2]) == [1, 2]
    assert ping_pong([1]) == [1]
    # Mirroring must not repeat the end frame, or the loop stutters there.
    assert ping_pong([1, 2, 3])[-1] != ping_pong([1, 2, 3])[0]
    print("OK")


if __name__ == "__main__":
    selfcheck() if "--self-test" in sys.argv else main()
