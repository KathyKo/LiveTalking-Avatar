# RunPod Ditto Setup

Use this after a pod restart. Run commands inside the pod, not on Windows.

## 1. Activate Env

Default venv path is `/root/ditto310`. Do not put the venv in `/workspace` if pip shows `[Errno 5] Input/output error`.

If the env is missing, rebuild it first:

```bash
export DITTO_VENV=/root/ditto310
apt-get update
apt-get install -y python3.10 python3.10-venv ffmpeg libegl1 libgles2 libgl1

cd /workspace
python3.10 -m venv "$DITTO_VENV"
source "$DITTO_VENV/bin/activate"
python -m pip --version
```

Normal restart, when the env already exists:

```bash
export DITTO_VENV=/root/ditto310
cd /workspace/LiveTalking
source "$DITTO_VENV/bin/activate"
```

## 2. Required System Packages

Only run if the pod is fresh or MediaPipe complains about EGL/GLES.

```bash
apt-get update
apt-get install -y ffmpeg libegl1 libgles2 libgl1
```

## 3. Required Pip Packages

```bash
python -m pip install \
  numpy==1.26.4 scipy==1.11.4 scikit-image==0.22.0 pillow==11.3.0 \
  onnxruntime-gpu==1.18.1 \
  opencv-python-headless==4.11.0.86 opencv-contrib-python==4.11.0.86 \
  filetype tqdm Cython cuda-python==12.8.0 \
  resampy librosa soundfile imageio imageio-ffmpeg einops pyyaml av \
  mediapipe insightface moviepy decorator ml_dtypes certifi \
  aiohttp aiohttp_cors aiortc edge-tts elevenlabs \
  transformers diffusers accelerate \
  flask flask-sockets gevent gevent-websocket configargparse
```

If pip hits `[Errno 5] Input/output error` in `/workspace`, do not rebuild. Install only missing modules into `/root/ditto_pydeps`:

```bash
mkdir -p /root/ditto_pydeps
python -m pip install --target /root/ditto_pydeps --upgrade \
  librosa mediapipe insightface moviepy aiohttp aiohttp_cors aiortc \
  edge-tts elevenlabs transformers diffusers
export PYTHONPATH=/root/ditto_pydeps:$PYTHONPATH
```

Install TensorRT separately:

```bash
find ${DITTO_VENV:-/root/ditto310}/lib/python3.10/site-packages -maxdepth 1 \( -name '-ip*' -o -name '~ip*' \) -exec rm -rf {} +
python -m pip install --no-build-isolation --extra-index-url https://pypi.nvidia.com tensorrt==8.6.1
```

If `pip` says `pyximport` is missing, install Cython. `pyximport` is not a package name.

```bash
python -m pip install Cython -q
```

## 4. ASR

```bash
python -m pip install funasr modelscope sentencepiece --no-cache-dir
```

If `modelscope_hub ... METADATA` is broken:

```bash
python -m pip uninstall -y modelscope modelscope-hub funasr
rm -rf ${DITTO_VENV:-/root/ditto310}/lib/python3.10/site-packages/modelscope*
rm -rf ${DITTO_VENV:-/root/ditto310}/lib/python3.10/site-packages/funasr*
rm -rf ${DITTO_VENV:-/root/ditto310}/lib/python3.10/site-packages/modelscope_hub*
python -m pip install modelscope funasr sentencepiece --no-cache-dir
```

## 5. Runtime Library Path

Run before starting the app.

```bash
export LD_LIBRARY_PATH=${DITTO_VENV:-/root/ditto310}/lib/python3.10/site-packages/nvidia/cudnn/lib:${DITTO_VENV:-/root/ditto310}/lib/python3.10/site-packages/nvidia/cublas/lib:${DITTO_VENV:-/root/ditto310}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${DITTO_VENV:-/root/ditto310}/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH
```

Check:

```bash
python - <<'PY'
import ctypes, torch, onnxruntime as ort
ctypes.CDLL("libcudnn.so.8")
import tensorrt as trt
print("torch", torch.__version__, torch.cuda.is_available())
print("ort", ort.__version__, ort.get_available_providers())
print("trt", trt.__version__)
PY
```

Expected providers include `CUDAExecutionProvider`.

## 6. Ditto Source Files

Use this layout:

```text
/workspace/LiveTalking/data/avatars/ditto_woman/
  idle.mp4
  source.mp4
```

Important:

- `source.mp4` should be a neutral/closed-mouth or light-smile video.
- Do not use a talking/open-mouth video as `source.mp4`; Ditto will generate a new mouth on top of it and the mouth/teeth can become huge.
- If `source.png` is used, speech can jump from moving `idle.mp4` to a still image. Use `source.mp4` for a smooth transition.
- Best simple setup: make `source.mp4` and `idle.mp4` the same neutral idle video.

If you accidentally used a talking video as source:

```bash
cd /workspace/LiveTalking/data/avatars/ditto_woman
mv source.mp4 source_talking_bad.mp4
cp idle.mp4 source.mp4
```

Official Ditto docs show inference with `--source_path ./example/image.png`, but the official loader supports both images and videos. In LiveTalking, video source is better because it keeps the background/body moving during speech.

## 7. Required LiveTalking Patch Check

Confirm `use_d_keys` / `DITTO_EXP_SCALE` are NOT in the `sdk.setup()` call. That
knob flattened the mouth into raw source playback and was reverted; the working
("大牙") baseline uses Ditto's default motion keys. This grep must return nothing:

```bash
grep -n 'use_d_keys\|DITTO_EXP_SCALE' /workspace/LiveTalking/avatars/ditto_avatar.py | grep -v '#'
```

Shape the mouth with the official Ditto params instead (all opt-in env vars):
`DITTO_EMO`, `DITTO_FADE_TYPE`, `DITTO_OVERLAP`, `DITTO_SMO_K_D`, `DITTO_SMO_K_S`.

## 8. Start Ditto Woman

```bash
export DITTO_VENV=/root/ditto310
cd /workspace/LiveTalking
source "$DITTO_VENV/bin/activate"

DITTO_REPO=/workspace/ditto-talkinghead \
DITTO_CFG=/workspace/ditto-talkinghead/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
DITTO_DATA_ROOT=/workspace/ditto-talkinghead/checkpoints/ditto_trt_Ampere_Plus \
DITTO_STEPS=8 \
DITTO_MAX_SIZE=640 \
DITTO_ONLINE=1 \
DITTO_EMO=4 \
DITTO_START_BUFFER=0 \
DITTO_HOLD=0.25 \
python app.py \
  --model ditto \
  --avatar_id ditto_woman \
  --transport webrtc \
  --tts elevenlabs \
  --REF_FILE SEWXl8lPSO01tdGbWECX \
  --listenport 8010 \
  --fps 25
```

Do NOT add `DITTO_EXP_SCALE` — it was reverted (see section 7). First confirm the
mouth is generating again (the "大牙" look is expected here), THEN shape it.

If the mouth is too large / too much teeth, tune in this order (never `EXP_SCALE`):

1. `DITTO_EMO` — try `0` (neutral) instead of `4`.
2. `DITTO_SMO_K_D=5` `DITTO_SMO_K_S=15` — smooth driving/source motion.
3. `DITTO_FADE_TYPE=d0` `DITTO_OVERLAP=70` — steadier online-chunk blending.
4. Source frame mouth state: if `source.mp4` shows teeth/open mouth, any
   generation inherits it. Swap to a closed-mouth `source.png` or a small-mouth
   `source.mp4` (see section 6).

## 8b. Prove it is actually generating (not source playback)

Before tuning anything, confirm the writer frames differ from the source. Add
`DITTO_DEBUG=1` to the launch env, speak once, then check the logs and dumps:

```bash
DITTO_DEBUG=1 ...same launch as above...
ls /tmp/ditto_debug/          # frame_*.jpg (writer), source_*.jpg, match_*, pump_ditto_*
grep ditto-dbg <your log>
```

Read the per-frame log line `writer#NN best_src=.. full=.. mouth=.. prev=..`:

- `full≈0` on every frame → writer is replaying the source (no generation).
- `prev≈0` → frames are static.
- Real generation → small `full`/`aligned` but a clear `mouth` diff AND nonzero
  `prev`. `pump_ditto_*.jpg` should look identical to `frame_*.jpg` (proves the
  frames shown during speech are writer frames, not idle/source).

## 9. Smoke Test

Open the RunPod HTTP URL for port `8010`, click `Connect`, then test `Echo`.

Backend should show:

```text
offer sessionid=...
Connection state is connected
ditto render start
writer: ...it
```

If it shows `elevenlabs tts time` but `writer: 0it`, check `stream_pipeline_online.py` for worker exceptions.

