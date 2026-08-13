# Every DITTO_* value the server runs with, in one sourceable place.
#
# start.sh sources this, and so must anything that has to match the running
# server -- notably scripts/ditto_make_idle.py, whose whole job is to produce
# frames that match generated speech pixel for pixel:
#
#     cd /opt/livetalking && source docker/ditto-env.sh
#     python scripts/ditto_make_idle.py --avatar ditto_woman
#
# Exported vars in the shell win, so the RunPod template still overrides these.
# Do NOT copy these numbers anywhere else; a second copy will drift.

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
DITTO_ROOT=${DITTO_ROOT:-/opt/ditto-talkinghead}

OLD_MODEL_ROOT="$WORKSPACE_ROOT/ditto-talkinghead/checkpoints"
if [[ -n "${DITTO_CHECKPOINTS:-}" ]]; then
    MODEL_ROOT="$DITTO_CHECKPOINTS"
elif [[ -f "$OLD_MODEL_ROOT/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" && \
        -f "$OLD_MODEL_ROOT/ditto_trt_Ampere_Plus/warp_network_fp16.engine" ]]; then
    MODEL_ROOT="$OLD_MODEL_ROOT"
else
    MODEL_ROOT="$WORKSPACE_ROOT/models/ditto"
fi
export MODEL_ROOT

export DITTO_REPO=${DITTO_REPO:-$DITTO_ROOT}
export DITTO_CFG=${DITTO_CFG:-$MODEL_ROOT/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl}
export DITTO_DATA_ROOT=${DITTO_DATA_ROOT:-$MODEL_ROOT/ditto_trt_Ampere_Plus}

# Rendering. These four decide what a generated frame looks like, so idle.mp4
# must be regenerated whenever any of them changes.
export DITTO_STEPS=${DITTO_STEPS:-5}
export DITTO_MAX_SIZE=${DITTO_MAX_SIZE:-896}
export DITTO_EMO=${DITTO_EMO:-0}
export DITTO_EXP=${DITTO_EXP:-0.85}
export DITTO_ONLINE=${DITTO_ONLINE:-1}
export DITTO_SMO_K_D=${DITTO_SMO_K_D:-1}

# Playback timing. Does not affect how a frame is rendered.
export DITTO_FEED_CAP=${DITTO_FEED_CAP:-20}
export DITTO_START_BUFFER=${DITTO_START_BUFFER:-6}
export DITTO_HOLD=${DITTO_HOLD:-0.10}
export DITTO_TAIL_MS=${DITTO_TAIL_MS:-500}
export DITTO_AV_OFFSET_MS=${DITTO_AV_OFFSET_MS:-60}
# 220 = 60ms hold (covers the DITTO_AV_OFFSET_MS audio backlog) + 160ms blend.
# 370 froze the last generated frame for 210ms before the blend even started,
# which was the visible pause at the end of every sentence.
export DITTO_FINAL_HOLD_MS=${DITTO_FINAL_HOLD_MS:-220}

export ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-0.6B}
export ASR_DEVICE=${ASR_DEVICE:-cuda:0}
