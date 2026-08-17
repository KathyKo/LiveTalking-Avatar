"""Serialize heavyweight CUDA model initialization within this process."""

from threading import RLock


# TensorRT and PyTorch both create CUDA state while their models are loaded.
# Initializing them concurrently from request and warm-up threads can stall in
# native code, so all model constructors share this re-entrant lock.
GPU_INIT_LOCK = RLock()
