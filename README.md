# GeoVerifyTrack

## Environment and dependencies

```bash
python3.12 -m venv .envs/trackon312
.envs/trackon312/bin/pip install -r environments/pylock.trackon-full.offline.txt
python3.11 -m venv .envs/track4world311
.envs/track4world311/bin/pip install -r environments/pylock.track4world-full.offline.txt
```

Use CUDA-enabled PyTorch and keep the versions defined by the lock files.

## Checkpoint placement

Download and verify the large geometry checkpoint with the pinned asset lock:

```bash
.envs/trackon312/bin/python scripts/download_verified.py \
  --url https://huggingface.co/TencentARC/Track4World/resolve/main/track4world_da3.pth \
  --output weights/track4world/track4world_da3.pth \
  --sha256 6c589c0f41632a2329920087dcd8cbd6465d4198d574f0d4186eeada78c492c7
```

Place the remaining checkpoints at:

```text
weights/trackon2_dinov2/trackon2_dinov2_checkpoint.pt
weights/trackon_r/verifier.pt
weights/track4world/track4world_da3.pth
runs/v13-overall-reproduction-403/spatial_seed0/calibrator.pt
runs/v13-overall-reproduction-403/visibility_seed0/calibrator.pt
runs/memory-repair-415/train_seed0/best.pt
```

## Dataset and preprocessing

Training uses 111 general video sources and 28 validation sources. Final evaluation uses 21 robot videos. Prepared data is stored in:

```text
runs/memory-repair-413/sequence_cache/
runs/memory-repair-413/coordinate_cache/
runs/memory-repair-415/verified_cache/
```

Each manifest must contain video paths, frame indices, query points and the train/validation split. Prepare verification features with:

```bash
.envs/trackon312/bin/python runs/memory-repair-415/prepare_verification.py
```

## Training

```bash
.envs/trackon312/bin/python runs/memory-repair-415/train.py \
  --seed 0 --epochs 3 --output runs/memory-repair-415/train_seed0
```

## Inference

Verify downloaded assets first:

```bash
.envs/trackon312/bin/python scripts/check_assets.py --root .
```

Run inference on a selected video:

```bash
.envs/trackon312/bin/python scripts/run_delivery.py \
  --output runs/verified-memory-output \
  --dataset-manifest runs/robot-development-expanded-002/manifest.json \
  --indices 0
```

## Evaluation

```bash
.envs/trackon312/bin/python scripts/run_delivery.py \
  --output runs/verified-memory-eval \
  --dataset-manifest runs/robot-development-expanded-002/manifest.json \
  --split dev --indices 0,1,2,3,4,5,6,7,8,9,10,11
```

## Core architecture

Robot videos contain asynchronous 3D estimates that may be unreliable. Directly injecting an incorrect estimate into a recurrent tracker can amplify a short error over time. Our system takes an RGB video and first-frame query points, builds a continuous 2D reference, and treats each asynchronous 3D estimate as a candidate state update. A historical-window verifier compares appearance and geometric trajectories; only supported evidence can update the bounded temporal memory. An independent reference state anchors later decisions. The system outputs frame-wise 2D tracks, visibility, reliability scores and associated 3D motion.

## Core insight

3D geometry is not accepted as an unconditional input: it must first explain the observed historical appearance, and only then is it allowed to modify future tracking memory.

