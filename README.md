# UIE4_VN

UIE4_VN is a self-contained, auditable LSUI underwater-image-enhancement research framework. Its twelve versions study INR type, position, topology, and backbone under a fixed LSUI protocol.

Versions v1-v6 preserve their original isolated implementations. The pre-INR variants v7-v10 use one thin shared composition wrapper and directly reuse the already-audited v1/v4 backbone and v2/v3 INR classes, so their mathematical implementations cannot drift.

## Controlled variants

### Bottleneck-INR

| Backbone | Baseline | Point-INR | GL-INR |
|---|---|---|---|
| NAF encoder/decoder | v1 | v2 | v3 |
| Plain U-Net | v4 | v5 | v6 |

### Pre-INR

| Backbone | Baseline | Pre-Point-INR | Pre-GL-INR |
|---|---|---|---|
| Plain U-Net | v4 | v7 | v8 |
| NAF encoder/decoder | v1 | v9 | v10 |

### UICF-INR placement

| Backbone | Baseline | Pre-backbone | Parallel correction branch |
|---|---|---|---|
| NAF encoder/decoder | v1 | v11 | v12 |

Within each matrix row, the ablation isolates the specified INR condition. v1-v3 use the same intro, NAF encoder stages, downsampling, decoder stages, upsampling, skip connections, ending convolution, and global image residual. Their only structural difference is `Identity` versus `Point-INR` versus `GL-INR`. The formal configurations intentionally set `middle_blk_num: 0`: there are no NAF middle blocks before or after the experimental bottleneck.

v4-v6 use the exact same classic four-level Plain U-Net backbone with Conv-BatchNorm-ReLU DoubleConv blocks, max-pooling, transposed-convolution upsampling, concat skip connections, and a direct sigmoid RGB output. v5 applies the unchanged v2 Point-INR and v6 applies the unchanged v3 GL-INR to the 1024-channel U-Net bottleneck feature, directly before the first decoder upsampling. Both INR modules keep their internal residual; the composition adds no second outer residual. These versions use the same LSUI split and training/evaluation protocol to measure INR × backbone interaction. **They are controlled experiments, not additional proposed architectures.**

v7-v10 instead apply exactly one INR before the backbone: `RGB [B,3,H,W] -> INR(channels=3) -> unchanged backbone -> RGB`. v7/v8 use the v4 Plain U-Net; v9/v10 use the v1 NAF encoder/decoder with its identity bottleneck. No adapter, clamp, normalization, intermediate INR, or extra output activation is inserted. Zero-initialized INR correction layers make each pre-INR model initially identical to its corresponding baseline when backbone states match.

v11/v12 use one canonical Underwater Implicit Correction-Field INR (UICF-INR) and the unchanged v1 NAF backbone. UICF predicts a full-resolution unconstrained field `R` and reconstructs exactly `I_uicf = I + R * (I - b)`. v11 computes `NAFNet(I_uicf)`. v12 is a parameter-free image-level fusion topology computing `NAFNet(I) + (I_uicf - I)`; it has no concatenation, projection, attention, gate, alpha, or fusion network. Both reduce exactly to v1 at initialization because the correction MLP output layer is zero-initialized.

Point-INR is a deliberately named feature-conditioned, absolute-coordinate baseline, not a claim of line-by-line reproduction of another INR paper. It concatenates each input feature vector with Fourier-encoded pixel-center coordinates, predicts a same-shaped residual with a chunked MLP, and adds it to the module input.

GL-INR projects its input tensor to a stride-2 latent grid. Its local branch queries four bounded geometric neighbors with one shared MLP and bilinearly ensembles the resulting implicit features. Its separate global branch reads only Fourier-encoded absolute coordinates. A fusion MLP maps concatenated local/global features back to the module input channel count. Phase one intentionally has no local Fourier encoding, cell decoding, unfolding, attention, physical prior, RGB head, GAN, diffusion, or vertical stack.

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
python -m src.v5.train --config configs/config_v5.yaml --seed 3520 --gpu 0
python -m src.v6.train --config configs/config_v6.yaml --seed 3520 --gpu 1
python -m src.v7.train --config configs/config_v7.yaml --seed 3520 --gpu 0
python -m src.v8.train --config configs/config_v8.yaml --seed 3520 --gpu 1
python -m src.v9.train --config configs/config_v9.yaml --seed 3520 --gpu 0
python -m src.v10.train --config configs/config_v10.yaml --seed 3520 --gpu 1
python -m src.v11.train --config configs/config_v11.yaml --seed 3520 --gpu 0
python -m src.v12.train --config configs/config_v12.yaml --seed 3520 --gpu 1

python -m src.v3.train \
  --config configs/config_v3.yaml \
  --seed 1234 \
  --gpu 0 \
  --data-root /root/autodl-tmp/pro/publicdata/LSUI19_dup_train \
  --name NAFEncDec_GLINR
```

All train entry points support `--config`, `--seed`, `--gpu`, `--data-root`, `--name`, and `--resume`; the Plain U-Net-derived v4-v8 entry points additionally support a `--batch-size` override. CUDA is selected from the requested index rather than hard-coded. CPU is supported for smoke checks; CUDA AMP disables itself on CPU. `training.deterministic: true` enables deterministic PyTorch behavior with explicit warnings for unsupported operations.

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
python -m src.v5.test --run-dir experiments/<v5_run> --checkpoint best_psnr --gpu 0
python -m src.v6.test --run-dir experiments/<v6_run> --checkpoint best_psnr --gpu 1
python -m src.v7.test --run-dir experiments/<v7_run> --checkpoint best_psnr --gpu 0
python -m src.v8.test --run-dir experiments/<v8_run> --checkpoint best_psnr --gpu 1
python -m src.v9.test --run-dir experiments/<v9_run> --checkpoint best_psnr --gpu 0
python -m src.v10.test --run-dir experiments/<v10_run> --checkpoint best_psnr --gpu 1
python -m src.v11.test --run-dir experiments/<v11_run> --checkpoint best_psnr --gpu 0
python -m src.v12.test --run-dir experiments/<v12_run> --checkpoint best_psnr --gpu 1
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
python tools/print_model_info.py --config configs/config_v5.yaml
python tools/print_model_info.py --config configs/config_v6.yaml
python tools/print_model_info.py --config configs/config_v7.yaml
python tools/print_model_info.py --config configs/config_v10.yaml
python tools/print_model_info.py --config configs/config_v11.yaml
python tools/print_model_info.py --config configs/config_v12.yaml

# Side-by-side completed or partial runs; missing test results print N/A
python tools/compare_runs.py experiments/<v1_run> experiments/<v2_run> experiments/<v3_run> experiments/<v4_run> experiments/<v5_run> experiments/<v6_run> experiments/<v7_run> experiments/<v8_run> experiments/<v9_run> experiments/<v10_run> experiments/<v11_run> experiments/<v12_run>

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

v4-v6 instead use the standard Plain U-Net channel path `64→128→256→512→1024→512→256→128→64`, four max-pooling stages, transposed-convolution upsampling, concat skips, and sigmoid output. They pad arbitrary inputs to a multiple of 16 and crop the direct prediction back to the original size. They have no global image residual and are intentionally not parameter-matched to v1-v3. v4 sends the 1024-channel bottleneck directly to the decoder; v5/v6 replace that identity path with Point-INR/GL-INR while preserving the same tensor shape.

The v7-v10 pre-INR modules always receive and return three-channel tensors at the original input resolution. Point-INR therefore has the same 20,739 parameters in v7 and v9; GL-INR has the same 139,651 parameters in v8 and v10. Their residual outputs are passed directly into the backbone without clamping.

UICF-INR has its own full-resolution 48-channel two-block image encoder, fixed eight-band periodic spatial encoding without a pi multiplier, global-average chromatic anchor `48→64→3→Sigmoid`, and a three-hidden-layer `128→128` per-pixel correction MLP. It has 137,734 parameters. Consequently v11 and v12 each have 1,115,689 parameters, exactly 137,734 more than v1; v12 adds no fusion parameters. Dataset tensors already use RGB float `[0,1]`, so no UICF domain adapter is used. UICF and final model forward paths do not clamp outputs; the existing validation/test protocol still clamps a detached float prediction only for metrics and PNG output.

Training uses synchronized paired 256×256 random crops, horizontal/vertical flips and 90-degree rotations. Small pairs are reflect-padded. Validation/test are deterministic and default to paired 256×256 bilinear resizing; set `evaluation.resize: false` for native-resolution evaluation. The same Charbonnier objective, AdamW settings, metric implementation, initialization, AMP behavior, and checkpoint protocol apply to every version.

Point/GL queries are chunked by `query_chunk` to bound MLP query memory. This limits the implicit-query intermediates, not the convolutional backbone activation memory.

## Version isolation

Do not refactor v1-v6 in ways that change existing behavior or checkpoint keys. New isolated experiments may copy a stable version; controlled cross-backbone experiments may instead use a small shared composition layer when direct reuse is necessary to guarantee identical backbone/INR implementations.

## Git policy

The fixed manifests, configs, source, tests, tools, README, and requirements belong in Git. LSUI images, experiment outputs, and all `.pt/.pth/.ckpt` files do not. `.gitignore` preserves only `experiments/.gitkeep` from the experiment directory.
