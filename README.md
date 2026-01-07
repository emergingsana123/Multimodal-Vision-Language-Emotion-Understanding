# Multimodal Vision-Language Emotion Understanding

Temporal valence/arousal regression on the AFEW-VA video dataset using CLIP image encoders, LoRA adapters, a lightweight temporal head, and staged contrastive pretraining plus regression fine-tuning. An alternative pipeline (Path B) pre-extracts CLIP features to speed up training on limited GPUs.

## Repository Layout
- [task2.2](task2.2): Main two-stage pipeline (contrastive pretrain then regression fine-tune) with CLIP + LoRA and temporal head.
- [task2_pathB](task2_pathB): Feature-extraction-first pipeline optimized for ~20GB GPUs; trains only a temporal transformer head on cached CLIP features.
- [outputs](outputs): Plots, metrics, and evaluation artifacts (created at runtime).
- Legacy/experiments: top-level `task2_*` scripts and `supplementary_analysis.py` are kept for reference.

## Environment Setup
1) Python 3.9+ and a CUDA GPU are recommended.
2) Install dependencies (base + task2.2 extras):
   ```bash
   pip install -r requirements.txt
   pip install -r task2.2/requirements.txt
   ```
3) Ensure ffmpeg is available if you plan to decode videos yourself (frames are expected as images here).

## Dataset Preparation (AFEW-VA)
Expected layout under a dataset root (defaults to `dataset/` under the repo; override via `--dataset` or config):
```
dataset/
  001/
    00000.png
    00001.png
    ...
    001.json   # frame-level valence/arousal annotations
  002/
    ...
```
Place the root directory anywhere and pass the path with `--dataset` (Pipeline A) or adjust `dataset_dir` in [task2_pathB/config.py](task2_pathB/config.py).

## Pipeline A: Full CLIP + LoRA Training (task2.2)
Scripts live in [task2.2](task2.2). Config defaults are in [task2.2/config.py](task2.2/config.py); override via CLI flags.

Common overrides: `--dataset`, `--batch_size`, `--num_frames`, `--epochs_pretrain`, `--epochs_finetune`, `--use_lora/--no-use_lora`, `--checkpoint`.

1) Contrastive pretraining
   ```bash
   cd task2.2
   python main.py --stage pretrain --dataset /path/to/AFEW-VA
   ```
   - Saves checkpoints to `checkpoints/pretrain_*.pth` and history to `outputs/pretrain_history.json`.

2) Regression fine-tuning
   ```bash
   python main.py --stage finetune --dataset /path/to/AFEW-VA \
     --checkpoint checkpoints/pretrain_best.pth
   ```
   - Produces `checkpoints/finetune_*.pth` and `outputs/finetune_history.json`.

3) Full run (pretrain + finetune in one go)
   ```bash
   python main.py --stage all --dataset /path/to/AFEW-VA
   ```

4) Evaluation only
   ```bash
   python main.py --stage evaluate --dataset /path/to/AFEW-VA \
     --checkpoint checkpoints/finetune_best_best.pth
   ```
   - Writes metrics and plots to `outputs/evaluation/` and a submission CSV.

Key components: dataset loader with temporal sampling ([task2.2/dataset.py](task2.2/dataset.py)), CLIP + LoRA + temporal/regression heads ([task2.2/model.py](task2.2/model.py)), contrastive losses ([task2.2/losses.py](task2.2/losses.py)), training loops ([task2.2/train_pretrain.py](task2.2/train_pretrain.py), [task2.2/train_finetune.py](task2.2/train_finetune.py)), evaluation/visualization ([task2.2/evaluate.py](task2.2/evaluate.py), [task2.2/utils.py](task2.2/utils.py)).

## Pipeline B: Cached-Feature Temporal Head (task2_pathB)
Faster alternative when GPU memory or time is tight. Config in [task2_pathB/config.py](task2_pathB/config.py).

1) Extract CLIP features once
   ```bash
   cd task2_pathB
   python 1_extract_features.py --dataset_dir /path/to/AFEW-VA   # uses config default if omitted
   ```
   - Saves per-clip features.npy + metadata and builds `features_cache/index.json`.

2) Train temporal transformer head on cached features
   ```bash
   python 5_train.py
   ```
   - Uses contrastive InfoNCE with in-batch negatives; logs to `outputs/logs`, checkpoints to `outputs/checkpoints`.

3) Resume
   ```bash
   python 5_train.py --resume
   ```

## Outputs
- Checkpoints: `checkpoints/` (Pipeline A) or `task2_pathB/outputs/checkpoints/` (Pipeline B).
- Metrics/plots: `outputs/` in each pipeline. Evaluation also writes `evaluation_results.json` and `submission.csv`.
- Feature cache (Path B): `task2_pathB/outputs/features_cache/`.

## Tips
- GPUs: Both pipelines auto-detect CUDA; CPU is supported but slow. Mixed precision is enabled by default where safe.
- Dataset sanity: `python task2.2/main.py --stage pretrain --dataset ...` prints clip counts and warns if frames/JSON are missing.
- Reproducibility: Seeds are set in both pipelines; pass `--seed` (Pipeline A) or edit config (Pipeline B) to change.
