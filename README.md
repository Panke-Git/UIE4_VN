# UIE4_VN

UIE4_VN is a self-contained, auditable LSUI underwater-image-enhancement research framework. Versions v1-v3 answer one controlled question: when the encoder and decoder are fixed, does a feature-level Global-Local Implicit Neural Representation outperform an identity bottleneck and a point-wise absolute-coordinate INR bottleneck? Version v4 is a separate architecture sanity baseline for the current dataset split and evaluation protocol.

The repository deliberately contains four isolated implementations. No version imports another version and there is no shared experiment-code package.

## Controlled variants

| Version | Bottleneck path | Everything else |
|---|---|---|
| v1 | `E -> Identity(E) -> decoder` | Fixed NAF-style encoder/decoder, data, loss, optimizer and protocol |
| v2 | `E -> Point-INR(E, abs-coord) -> decoder` | Identical to v1 |
| v3 | `E -> GL-INR(E, abs/relative-coord) -> decoder` | Identical to v1 |

The current ablation isolates the bottleneck representation. All three variants use the same intro, encoder stages, downsampling, decoder stages, upsampling, skip connections, ending convolution, and global image residual. Their only structural difference is `Identity` versus `Point-INR` versus `GL-INR`. The formal configurations intentionally set `middle_blk_num: 0`: there are no NAF middle blocks before or after the experimental bottleneck.

The independent v4 baseline is a classic four-level Plain U-Net with Conv-BatchNorm-ReLU DoubleConv blocks, max-pooling, transposed-convolution upsampling, concat skip connections, and a direct sigmoid RGB output. It uses the same LSUI split and training/evaluation protocol to test whether strong results are specific to the NAF-based backbone. **v4 is not a proposed model and is not part of GL-INR.**

Point-INR is a deliberately named feature-conditioned, absolute-coordinate baseline, not a claim of line-by-line reproduction of another INR paper. It concatenates each bottleneck feature with Fourier-encoded pixel-center coordinates, predicts a same-shaped feature residual with a chunked MLP, and adds it to `E`.

GL-INR projects `E` to a stride-2 latent grid. Its local branch queries four bounded geometric neighbors with one shared MLP and bilinearly ensembles the resulting implicit features. Its separate global branch reads only Fourier-encoded absolute coordinates. A fusion MLP maps concatenated local/global features back to the bottleneck channel count. Phase one intentionally has no local Fourier encoding, cell decoding, unfolding, attention, physical prior, RGB head, GAN, diffusion, or vertical stack.

## Fixed LSUI protocol

The committed manifests in `split/lsui19/` are the protocol and must not be regenerated:

- `train.tsv`: 3466 samples used for gradient training.
- `validation.tsv`: 385 samples used for validation and checkpoint selection.
- `test.tsv`: 428 held-out samples; test images and performance are accessed only by an explicit `src.v*.test` command.
- `train + validation = 3851`, corresponding to the original LSUI training portion.

The actual files have no header and exactly three tab-separated fields: `sample_id`, relative input path, relative GT path. Training paths begin with `Train/input` and `Train/GT`; test paths begin with `Val/input` and `Val/GT`.

On the target server the default root is:

```text
/root/autodl-tmp/pro/publicdata/LSUI19_dup_train
```

Expected paths are formed as `data.root / manifest_relative_path`. Change only `data.root` or pass `--data-root`; do not edit the manifests.

## Installation

Use Python 3.10 or newer. Install a PyTorch build appropriate for the server CUDA version from the official PyTorch instructions, then install the small remaining dependency set:

```bash
git clone <UIE4_VN repository URL>
cd UIE4_VN
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url <the index appropriate for your CUDA runtime>
pip install -r requirements.txt
python tools/validate_splits.py
python -m pytest -q
```

The repository does not depend on BasicSR, torchvision model implementations, PyTorch Lightning, Hydra, or an external experiment tracker.

## Train

Each module defaults to its own YAML. CLI values override YAML values.

```bash
python -m src.v1.train
python -m src.v2.train
python -m src.v3.train
python -m src.v4.train --gpu 0

python -m src.v3.train \
  --config configs/config_v3.yaml \
  --seed 1234 \
  --gpu 0 \
  --data-root /root/autodl-tmp/pro/publicdata/LSUI19_dup_train \
  --name NAFEncDec_GLINR
```

All train entry points support `--config`, `--seed`, `--gpu`, `--data-root`, `--name`, and `--resume`; v4 additionally supports a `--batch-size` override. CUDA is selected from the requested index rather than hard-coded. CPU is supported for smoke checks; CUDA AMP disables itself on CPU. `training.deterministic: true` enables deterministic PyTorch behavior with explicit warnings for unsupported operations.

Training validates all train/validation files before optimization. It reads test metadata only to audit fixed counts/leakage and snapshot the protocol; it never creates a test Dataset, opens test images, or uses test performance for selection. `test.auto_run_after_training` remains false.

## Resume

Resume appends to the original run and restores the model, optimizer, AMP scaler, epoch, best values, and available Python/NumPy/PyTorch/CUDA RNG states:

```bash
python -m src.v3.train \
  --resume experiments/<v3_run>/checkpoint/last.pt \
  --gpu 0
```

The saved resolved architecture and version must match. A changed server dataset location can be supplied with `--data-root`. Changing the seed or run name during resume is rejected.

## Validation and held-out test

Validation runs at `training.validate_every`, computes Charbonnier loss, float-RGB PSNR, and Gaussian-window float-RGB SSIM before PNG quantization, and writes the latest per-image CSV/summary. Independent best checkpoints are maintained for validation loss, PSNR, and SSIM.

Held-out testing is always explicit and reconstructs the model from the run's `config_resolved.yaml`. It uses the run's split snapshot, so later edits to repository manifests cannot silently change an old experiment.

```bash
python -m src.v1.test --run-dir experiments/<v1_run> --checkpoint best_psnr --gpu 0
python -m src.v2.test --run-dir experiments/<v2_run> --checkpoint best_psnr --gpu 0
python -m src.v3.test --run-dir experiments/<v3_run> --checkpoint best_psnr --gpu 0
python -m src.v4.test --run-dir experiments/<v4_run> --checkpoint best_psnr --gpu 0
```

Checkpoint selectors are `best_psnr`, `best_ssim`, `best_loss`, and `last`; an explicit checkpoint path is also accepted. Test allows `--gpu` and `--data-root` overrides but no architecture override. Outputs include all enhanced PNGs, per-image metrics, a summary, ten deterministic sample images, their fixed index manifest, and a 10×3 `input | enhanced | GT` grid.

## Experiment artifacts

New runs use `{version}_{name}_seed{seed}_{YYYYMMDD_HHMMSS}` beneath `experiments/` and contain:

```text
best/                 best_loss/psnr/ssim .pt and .json
checkpoint/           periodic epoch files and last.pt
log/                  train/val/test logs and JSON/CSV history
result/               validation and explicit-test metrics/images/grid
split_snapshot/       exact manifests plus hashes in run_info.json
config_source.yaml    input YAML before CLI overrides
config_resolved.yaml  exact effective YAML
config.json           exact effective JSON
run_info.json         environment, Git, data, device and model provenance
status.json           running/completed/failed state and best values
```

`status.json` is updated on every epoch and records exception type/message on failure; the full traceback goes to `train.log`. Checkpoints contain all training states, the resolved config, seed, best values, and RNG states.

## Utilities

```bash
# Manifest schema/count/hash/leakage checks; image checks run only when data.root exists
python tools/validate_splits.py
python tools/validate_splits.py --data-root /path/to/LSUI19_dup_train

# Model parameters and architecture-specific feature/output shapes
python tools/print_model_info.py --config configs/config_v3.yaml
python tools/print_model_info.py --config configs/config_v4.yaml

# Side-by-side completed or partial runs; missing test results print N/A
python tools/compare_runs.py experiments/<v1_run> experiments/<v2_run> experiments/<v3_run> experiments/<v4_run>

# Full model-free LSUI split/difficulty/duplicate diagnostic
python tools/diagnose_lsui.py --config configs/config_v1.yaml

# AutoDL data-root override (the CLI value takes precedence over YAML)
python tools/diagnose_lsui.py \
  --config configs/config_v1.yaml \
  --data-root /root/autodl-tmp/pro/publicdata/LSUI19_dup_train

# Post-hoc clean-test sensitivity analysis from existing per-image CSV metrics
python tools/analyze_clean_test.py \
  --diagnostic-dir diagnostics/<diagnostic_run> \
  --v1-run experiments/<identity_run> \
  --v2-run experiments/<point_inr_run> \
  --v3-run experiments/<glinr_run>

# Static and numerical verification
python -m compileall src
python -m pytest -q
```

The LSUI diagnostic reads the fixed TSVs without modifying them and does not run a
model. It compares raw Input→GT difficulty at the current paired 256×256 bilinear
evaluation size and, where native shapes match, at native resolution. It also
reports resolution and basic RGB/luminance/saturation distributions; raw-file,
decoded-pixel, and cross-split exact duplicates; and cross-split 64-bit dHash
near-duplicate candidates for manual inspection. Results are written beneath a
timestamped `diagnostics/lsui19_YYYYMMDD_HHMMSS/` directory. No source images are
copied there.

The clean-test sensitivity command is a post-hoc analysis. It reads the existing
diagnostic candidate tables and the three completed runs' float-output
`result/test_metrics.csv` files; it does not retrain models, load checkpoints,
rerun inference, or recompute metrics from saved PNG images. Outputs are written
only beneath a new timestamped `analysis/clean_test_YYYYMMDD_HHMMSS/` directory.

## Configuration and architecture

All experiment values live in YAML. Defaults are a 3-channel, width-32 NAF-style encoder/decoder with encoder blocks `[2,2,2]`, zero middle blocks, decoder blocks `[2,2,2]`, three downsamplings (factor 8), and a 256-channel bottleneck feature. The encoder output goes directly through the version-specific bottleneck and then into the decoder. The network pads arbitrary input height/width to the factor and crops its globally residual output back to the original shape.

v4 instead uses the standard Plain U-Net channel path `64→128→256→512→1024→512→256→128→64`, four max-pooling stages, transposed-convolution upsampling, concat skips, and sigmoid output. It pads arbitrary inputs to a multiple of 16 and crops the direct prediction back to the original size. It has no global image residual or advanced module and is intentionally not parameter-matched to v1-v3.

Training uses synchronized paired 256×256 random crops, horizontal/vertical flips and 90-degree rotations. Small pairs are reflect-padded. Validation/test are deterministic and default to paired 256×256 bilinear resizing; set `evaluation.resize: false` for native-resolution evaluation. The same Charbonnier objective, AdamW settings, metric implementation, initialization, AMP behavior, and checkpoint protocol apply to every version.

Point/GL queries are chunked by `query_chunk` to bound MLP query memory. This limits the implicit-query intermediates, not the NAFNet convolutional activation memory.

## Version isolation

Do not modify existing versions to share code. A future version should copy one complete stable version into its own `src/vN`, keep all imports version-relative, add its own config, and introduce only the specified experimental difference.

## Git policy

The fixed manifests, configs, source, tests, tools, README, and requirements belong in Git. LSUI images, experiment outputs, and all `.pt/.pth/.ckpt` files do not. `.gitignore` preserves only `experiments/.gitkeep` from the experiment directory.
