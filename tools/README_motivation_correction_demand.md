# Reference-derived correction demand: frozen analysis protocol

Protocol frozen on 2026-11-06 (Asia/Shanghai), after repository/data audit and
before calculating correction metrics or viewing selected examples.

## Audit evidence

- WSL repository: `/home/paxton/pro/UIE4_VN`; origin
  `https://github.com/Panke-Git/UIE4_VN.git`; branch `main`;
  HEAD `27ebf339ab83e0cf2b248324c2f89f7a68e9b681`.
- Existing untracked `tools/audit_uicf_experiments.py` must remain untouched.
- Read README, both V16 configs, V16 dataset/test implementation, all six TSVs,
  UIEB protocol, and the candidate run's resolved config/split snapshots.
- Canonical counts: LSUI 3466/385/428; UIEB 720/80/90. All manifests pass the
  project's uniqueness and leakage validator. All 518 test pairs pass the
  project's image validator. No missing, corrupt, non-RGB or mismatched pair.
- Roots: `/home/paxton/pro/publicdata/LSUI19_dup_train` and
  `/home/paxton/pro/publicdata/UIEB19`. Configuration roots refer to a different
  machine and are overridden explicitly; configs are never edited.
- LSUI test hash: `cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c`.
- UIEB test hash: `3693f31fdc849684706ad55f95e0c33e2b362082ced8346da033620856ffe07a`.
  UIEB T90 has 82 Train/input and 8 Val/input paths. Membership is TSV-defined.
- Candidate LSUI run snapshot hashes equal all three canonical LSUI manifests.
  Its evaluation config is the same 256-square bilinear protocol. No weights
  are opened, hashed, loaded, or copied.
- Detailed evidence and original-file SHA256s are in the timestamped
  `analysis/motivation_correction_demand/audit_*/repository_data_audit.json`.

## Fixed computation and eligibility

Use the existing `src.v16.dataset.read_manifest`, `validate_split_protocol`,
`verify_image_entries` functions. Importing that dataset module imports PyTorch
but does not create a model, initialize CUDA or access a checkpoint. Pair image
loading exactly reproduces its evaluation path: verify RGB and original paired
dimensions, PIL BILINEAR resize both images to 256x256, NumPy float32 / 255.
Promote those float32 values to float64 **before** residual subtraction and
reductions. No linear-light conversion, crop, augmentation, color enhancement,
or aspect-ratio preservation is added. Original-resolution dimensions are
recorded; the 256-square residual is not a native-resolution analysis.

For delta=Y-I and mu=mean_xy(delta), compute mean pixel L2 magnitude,
E_raw=mean(delta^2), E_global=mean((delta-mu)^2), eta=E_global/(E_raw+1e-12),
channel signed means/absolute means/spatial standard deviations,
D_channel=max(mu)-min(mu), and V_channel=population_variance(mu).
All variances and standard deviations use ddof=0. Ratios are null and flagged
degenerate when E_raw<=1e-10, rather than epsilon-dominated. Channel means
also provide paired R-G/R-B/G-B image-level differences.

Non-overlapping full patches of 8,16,32 pixels are averaged first. Calculate
V_patch=mean_c population_variance_p(patch_mean_delta). Require exact spatial
divisibility; never silently crop or pad. V_16/E_global (when nonzero) reports
the fraction of spatial residual energy retained after block averaging.
This is additional descriptive evidence for coarse spatial structure, not
proof that registration/reference artifacts have been eliminated.

Aggregate per-image numbers with equal image weights, separately for LSUI and
UIEB: mean, median, population std, linear-interpolated p05/p95 and valid/null
counts. Record failed pairs with paths and exception details. Never silently
replace a failing test member or substitute train/validation images.

## Frozen selection and display

After the entire requested test split is analyzed, select three distinct valid,
nondegenerate examples per dataset: V_16 rank nearest q=0.50; V_16 rank nearest
q=0.80; D_channel rank nearest q=0.80. Quantile rank is floor(q*(n-1)+0.5).
For a duplicate choice, use the nearest unused rank, preferring the lower rank
on ties. Sort by metric, sample_id (lexicographic), input path, GT path. Keep
selection JSON before plotting. Seed=3520; selection uses no random draws.
Unverified visual artifacts are flagged after inspection, never used for
unrecorded reselection. Automatic validity checks determine eligibility; no
additional metric outlier exclusion is applied.

Main: three rows of Input | Reference | ||delta||_2 | delta_R | delta_G |
delta_B. Identical image preprocessing and no per-panel normalization.
All datasets, all examples and all three signed channels share a linear,
zero-centered coolwarm range [-0.5,+0.5]. Magnitude uses linear magma [0,1].
These fixed limits are registered before measuring residual distributions.
Log both whole-test and selected-image display saturation fractions. Provide
an additional theoretical-range version with signed [-1,+1] and magnitude
[0,sqrt(3)] so display saturation cannot be the only source of the phenomenon.
Never change a numerical result for display. Colorbars state RGB [0,1] units.

Diagnostic: Input | GT-dependent Global RGB Oracle | Reference | Remaining
Correction Magnitude. Display clip I+mu only to [0,1]; compute the remaining
residual and all errors against the **unclipped** oracle. Record both scalar
and pixel clipping fractions. Separate patch maps show block-averaged residuals
at the main patch size. Quantitative distribution/sensitivity panels retain
the same dataset identities. Export PNG >=300 dpi, PDF and SVG with vector text.

## Execution, checks and interpretation

Only add the independent tool, its tests, this protocol and timestamped results.
Never edit source models/configs/splits or existing tools. Refuse collisions;
every invocation makes a fresh timestamped run directory. `--sample-limit`
always marks a run PARTIAL/smoke, even if its limit exceeds the split size.
PASS requires the entire fixed split, zero failed pairs, protocol and numerical
invariants, source stability checks and requested outputs. A nonzero descriptive
effect does not imply that the scientific hypothesis succeeded.

Run synthetic numeric/data/visual-consistency tests, then four real pairs per
dataset, then all 428 and 90 pairs. Independently recompute consequential
metrics from the original images with a separate calculation. Render PDF and
SVG exports and inspect labels, symbols, colorbars and row identities. Keep
logs and make ZIPs from task artifacts only, excluding datasets/checkpoints.

Report actual values, coarse-scale consistency and paired channel differences.
Discuss reference construction, registration, resized resolution and global
**additive** model restrictions. Do not extend the conclusion to arbitrary
global nonlinear transformations, superiority/necessity of INR, learned UICF
R(x), physical degradation fields, or SOTA rankings. Introduction suitability
is assessed after seeing the actual numerical and visual evidence.

## Commands

```bash
PY=/home/paxton/miniconda3/envs/pax/bin/python
$PY -m pytest -q tests/test_motivation_correction_demand.py
$PY tools/analyze_motivation_correction_demand.py --dataset lsui --config configs/config_v16.yaml --data-root /home/paxton/pro/publicdata/LSUI19_dup_train --sample-limit 4 --seed 3520 --patch-size 16 --export-zip
$PY tools/analyze_motivation_correction_demand.py --dataset uieb --config configs/uieb/config_v16_uieb.yaml --data-root /home/paxton/pro/publicdata/UIEB19 --sample-limit 4 --seed 3520 --patch-size 16 --export-zip
# Full runs use the same commands with --sample-limit omitted.
```
