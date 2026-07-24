#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/opt/livetalking
DITTO_ROOT=/opt/ditto-talkinghead
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
CACHE_ROOT=${CACHE_ROOT:-$WORKSPACE_ROOT/cache}
AVATAR_ID=${AVATAR_ID:-ditto_woman}
DEFAULT_AVATAR_ROOT=/opt/default-avatars

OLD_MODEL_ROOT="$WORKSPACE_ROOT/ditto-talkinghead/checkpoints"
if [[ -n "${DITTO_CHECKPOINTS:-}" ]]; then
    MODEL_ROOT="$DITTO_CHECKPOINTS"
elif [[ -f "$OLD_MODEL_ROOT/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" && \
        -f "$OLD_MODEL_ROOT/ditto_trt_Ampere_Plus/warp_network_fp16.engine" ]]; then
    MODEL_ROOT="$OLD_MODEL_ROOT"
else
    MODEL_ROOT="$WORKSPACE_ROOT/models/ditto"
fi

if compgen -G "$WORKSPACE_ROOT/LiveTalking/data/avatars/$AVATAR_ID/source.*" >/dev/null; then
    DATA_ROOT="$WORKSPACE_ROOT/LiveTalking/data"
else
    DATA_ROOT=${DATA_ROOT:-$WORKSPACE_ROOT/data}
fi

mkdir -p "$DATA_ROOT/avatars" "$MODEL_ROOT" "$CACHE_ROOT/modelscope" "$CACHE_ROOT/huggingface"
for bundled_avatar in "$DEFAULT_AVATAR_ROOT"/*; do
    [[ -d "$bundled_avatar" ]] || continue
    bundled_id=$(basename "$bundled_avatar")
    if ! compgen -G "$DATA_ROOT/avatars/$bundled_id/source.*" >/dev/null; then
        cp -a "$bundled_avatar" "$DATA_ROOT/avatars/$bundled_id"
        echo "Installed bundled avatar: $bundled_id"
    fi
done
ln -sfn "$DATA_ROOT" "$APP_ROOT/data"

JUPYTER_TOKEN=${JUPYTER_TOKEN:-$(python -c 'import secrets; print(secrets.token_urlsafe(24))')}
echo "JupyterLab: http://<pod-host>:8888/?token=$JUPYTER_TOKEN"
jupyter lab \
    --allow-root \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8888 \
    --ServerApp.root_dir="$WORKSPACE_ROOT" \
    --ServerApp.allow_remote_access=True \
    --IdentityProvider.token="$JUPYTER_TOKEN" &

export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}
export HF_HOME=${HF_HOME:-$CACHE_ROOT/huggingface}
export PYTHONPATH="$DITTO_ROOT:$APP_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
for lib_dir in nvidia/cuda_runtime/lib nvidia/cublas/lib nvidia/cudnn/lib nvidia/cufft/lib; do
    if [[ -d "$PY_SITE/$lib_dir" ]]; then
        export LD_LIBRARY_PATH="$PY_SITE/$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

if [[ -z "${ELEVENLABS_API_KEY:-}" ]]; then
    echo "ERROR: Set ELEVENLABS_API_KEY in the RunPod template." >&2
    exit 2
fi

while ! compgen -G "$DATA_ROOT/avatars/$AVATAR_ID/source.*" >/dev/null; do
    echo "Waiting for $DATA_ROOT/avatars/$AVATAR_ID/source.mp4 (upload it through JupyterLab)..."
    sleep 5
done

if [[ ! -f "$MODEL_ROOT/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" || \
      ! -f "$MODEL_ROOT/ditto_trt_Ampere_Plus/warp_network_fp16.engine" ]]; then
    echo "Ditto checkpoints are missing; downloading them once to $MODEL_ROOT ..."
    hf download digital-avatar/ditto-talkinghead \
        --include "ditto_cfg/*" "ditto_trt_Ampere_Plus/*" \
        --local-dir "$MODEL_ROOT"
fi

export DITTO_REPO="$DITTO_ROOT"
export DITTO_CFG=${DITTO_CFG:-$MODEL_ROOT/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl}
export DITTO_DATA_ROOT=${DITTO_DATA_ROOT:-$MODEL_ROOT/ditto_trt_Ampere_Plus}
export DITTO_STEPS=${DITTO_STEPS:-8}
export DITTO_MAX_SIZE=${DITTO_MAX_SIZE:-768}
export DITTO_ONLINE=${DITTO_ONLINE:-1}
export DITTO_EMO=${DITTO_EMO:-0}
export DITTO_SMO_K_D=${DITTO_SMO_K_D:-1}
export DITTO_EXP=${DITTO_EXP:-0.85}
export DITTO_FEED_CAP=${DITTO_FEED_CAP:-20}
export DITTO_START_BUFFER=${DITTO_START_BUFFER:-5}
export DITTO_HOLD=${DITTO_HOLD:-0.04}
export DITTO_TAIL_MS=${DITTO_TAIL_MS:-300}
export DITTO_AV_OFFSET_MS=${DITTO_AV_OFFSET_MS:-0}
export ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-0.6B}
export ASR_DEVICE=${ASR_DEVICE:-cuda:0}

cd "$APP_ROOT"
exec python app.py \
    --model ditto \
    --avatar_id "$AVATAR_ID" \
    --transport webrtc \
    --tts elevenlabs \
    --REF_FILE "${VOICE_ID:-SEWXl8lPSO01tdGbWECX}" \
    --listenport "${LISTEN_PORT:-8010}" \
    --fps 25 \
    "$@"
