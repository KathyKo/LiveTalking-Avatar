# RunPod Docker Deployment

This replaces the copied `ditto310` / `ditto_pydeps` environments. Do not run
`pip install` inside a pod created from this image.

## Build the image

Push `main` to GitHub. The `Build Ditto image` workflow publishes:

```text
ghcr.io/kathyko/livetalking-avatar:latest
```

The image contains LiveTalking, Python 3.10, PyTorch 2.5.1, CUDA libraries,
cuDNN, TensorRT 8.6.1, ONNX Runtime GPU, Qwen ASR dependencies, and the pinned
official Ditto source with the `np.arctan2` compatibility patch.

If the GitHub package is private, either make the package public or add GHCR
registry authentication to the RunPod template using a GitHub PAT with
`read:packages`.

## RunPod template

```text
Container image: ghcr.io/kathyko/livetalking-avatar:latest
Container disk:  30 GB
Volume disk:     50 GB or larger
Volume mount:    /workspace
HTTP port:       8010
```

Required environment variable:

```text
ELEVENLABS_API_KEY=<your rotated key>
```

Useful optional variables:

```text
AVATAR_ID=ditto_woman
VOICE_ID=SEWXl8lPSO01tdGbWECX
ASR_MODEL=Qwen/Qwen3-ASR-0.6B
ASR_DEVICE=cuda:0
DITTO_STEPS=15
DITTO_MAX_SIZE=768
DITTO_EMO=0
DITTO_EXP=0.85
```

For the male avatar:

```text
AVATAR_ID=ditto_man
VOICE_ID=aSXZu6bgEOS8MXVRzjPi
```

## Persistent files

Upload avatar files to either of these locations. The first path preserves
compatibility with the current pod layout:

```text
/workspace/LiveTalking/data/avatars/ditto_woman/source.mp4
/workspace/LiveTalking/data/avatars/ditto_woman/idle.mp4
```

or:

```text
/workspace/data/avatars/ditto_woman/source.mp4
/workspace/data/avatars/ditto_woman/idle.mp4
```

On the first start only, the container downloads the required Ditto folders to:

```text
/workspace/models/ditto
```

Qwen/ModelScope cache is stored in:

```text
/workspace/cache
```

Later pod starts reuse both directories and perform no pip installation.

## Open the app

Wait until the log shows:

```text
start http server; http://<serverip>:8010/index.html
```

Then open the RunPod HTTP URL for port `8010` with `/index-en.html`.

## Updating

Push code to `main`, wait for the GitHub workflow to finish, then restart the
pod using the new `latest` image. Models and avatars remain on `/workspace`.
