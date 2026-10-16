# V16 UICF correction-field intervention (frozen inference)

This is an **observational intervention on a frozen model**, not a new model,
training run, physical causal-identification experiment, or retrained ablation.
No result or degradation is assumed in advance. All reported numbers come from
actual float predictions on the complete frozen test manifest.

## Model and intervention point

Only canonical `src.v16` checkpoints are accepted:

```text
I -> original UICF -> (R, original b, original Ic) -> full Color-Query U-Net -> Dout
                     replace R only              -> same frozen backbone -> Dout_variant
```

`R` is an unconstrained signed coefficient field, **not** an RGB residual or a
physical transmission map. The reconstruction is exactly

```python
Ic_variant = I + R_variant * (I - b[:, :, None, None])
Dout_variant = model.backbone(Ic_variant)
```

Never normalize/abs/sigmoid/clip `R` or clip `Ic` before the backbone. Only final
predictions are converted with `.float().clamp(0,1)` for native metrics and PNGs.
The entire progressive color-query decoder, guidance and local refinement run
normally; tokens, network parameters and BN statistics are never edited.

One `forward_with_uicf_details(I)` supplies the **same R and b for all groups**.
Its original prediction and `details.enhanced` are reused without an extra
original reconstruction/backbone pass. A mandatory first-successful-image audit
also calls independent `model(I)` and `model.backbone(details.enhanced)` under
the same autocast. This independent normal-forward audit necessarily runs UICF
once more; its R/b are discarded, never used by any intervention. Set
`--consistency-every N` for additional predeclared audit positions.

## Fixed groups (no result-dependent selection)

| Group | Operation on original `[B,3,H,W]` R | Placement |
|---|---|---|
| original | unchanged | main table |
| spatial_mean | per-image, per-channel spatial mean, broadcast over H,W | main table |
| shift_xy | positive `torch.roll` by `(max(1,H//4), max(1,W//4))` on `(-2,-1)` | main table |
| channel_cycle_1 | new RGB receives old `[G,B,R]`, indices `[1,2,0]` | main table |
| shift_y | roll by `(max(1,H//4),0)` | supplementary |
| shift_x | roll by `(0,max(1,W//4))` | supplementary |
| channel_cycle_2 | new RGB receives old `[B,R,G]`, indices `[2,0,1]` | supplementary |
| zero_field | `zeros_like(R)`, so `Ic=I` | supplementary deactivation diagnostic |

All eight groups are always run. No best shift/permutation is selected. H and W
must both be >=2: otherwise nonzero registered spatial controls are impossible
and the sample fails. Actual shifts in pixels and fractions are recorded per
image. Rolls preserve each channel's coefficient multiset (checked), but **do not
preserve the induced correction** `R_variant*(I-b)`. No multi-shift aggregate is
implemented, so shifts are never treated as extra independent images.

## Data, checkpoints and metrics

- Read model/metric/evaluation configuration only from `RUN/config_resolved.yaml`.
- Require `experiment.version == 'v16'`, checkpoint `version == 'v16'`, matching
  checkpoint architecture and `strict=True` loading via the existing V16 helper.
  When checkpoint dataset identity/count metadata is present, it must also match
  the frozen run; a data-root relocation does not change dataset identity.
  Canonical UICF conditioning/anchor must be enabled, not a field-control ablation.
- Use **only** `RUN/split_snapshot/{train,validation,test}.tsv`. Call unchanged
  `validate_split_protocol` for configured counts, unique IDs/paths and leakage.
  Only test images are opened. No fallback to current project manifests; live
  manifest symlinks are rejected rather than treated as frozen snapshots.
- Modern LSUI19/UIEB identities and unambiguous legacy root/manifest metadata are
  resolved by the existing alignment helper without editing old configurations.
  Counts come from the snapshot and configured expected counts, never from a
  hardcoded dataset-size identity guess.
- The usual LSUI protocol is 3466/385/428 and UIEB T90 is 720/80/90. Other complete
  three-set V16 snapshots are honored according to their own frozen metadata.
- `src/v16_19` has a different Train/Val protocol and **no independent test.tsv**.
  Its Val must not be relabeled as this experiment's independent test.
- Use the exact test `LSUIDataset(..., split='test')`, PIL RGB bilinear resize
  (normally 256x256 for both input and GT), batch size **1**, native collate memory
  layout and original crop border. Native-resolution runs remain native when
  their frozen `evaluation.resize` is false. There is no resize CLI override.
- Reuse `src.v16.metrics.batch_metrics`, `src.shared.e00.batch_delta_e00` and
  `e00_protocol`. PSNR/SSIM/DeltaE00 are measured from final float RGB before PNG
  quantization. E00 uses sRGB -> Lab D65/2 degrees, CIEDE2000 kL=kC=kH=1,
  spatial pixel mean followed by equal per-image weighting. Lower E00 is better.
- AMP defaults to frozen `training.amp`, CUDA only; the same context covers UICF
  and **every** backbone call. Metrics run outside autocast. There is no automatic
  precision fallback. A non-finite original/intervention/metric fails the run.
- Preserve the process's backend/TF32 settings and record them. No silent switch
  to deterministic kernels, CPU, FP32 or a different batch size for a failing
  sample. CPU is naturally selected only when CUDA is unavailable, as in V16.
- Shapes/dtypes/devices/finite values, formula, field invariants, frozen state
  digest (including all buffers), Torch RNG and protected file hashes are audited.
  Pure interventions support float16/float32; production inputs remain native FP32.

### Equivalence and the old official results

The formula gate reuses `assert_uicf_consistency`: atol=1e-6, rtol=1e-5.
Normal/details/backbone/original forward comparisons use FP32 atol=1e-6,
rtol=1e-5; CUDA AMP atol=1e-5, rtol=1e-4, on **unclamped** outputs. Max errors
and checked sample IDs are recorded. Exceeding a gate fails, not a warning.

If `result/test_summary.json` and `test_metrics.csv` exist, check checkpoint
epoch/selector, sample identities/filenames/count, E00 protocol, and available
checkpoint/manifest hashes, resize/metric/AMP metadata. Known mismatches are
explicitly `NOT_COMPARABLE` with reasons. Matching legacy summaries do not
contain enough provenance to prove the historical AMP/resize/metric settings;
they are marked `NUMERIC_MATCH_LEGACY_METADATA_INCOMPLETE`, **not** a fully
verified protocol PASS. In the absence of a known protocol mismatch, numerical
discrepancies conservatively fail even for legacy metadata. Nothing edits old
files to fabricate a provenance match.

Per-image and overall absolute tolerances (PSNR, SSIM, E00) are
`(1e-5,1e-6,1e-5)` for FP32 and `(1e-3,1e-4,1e-3)` for CUDA AMP. The audit has
its own JSON and records actual observed errors; a different checkpoint or
explicit AMP override is not falsely claimed to reproduce old results.

## Run on AutoDL

Upload the new tool, `tools/uicf_intervention/` package and this README. No new
dependencies or downloads are required beyond the existing project environment.
`scikit-image` is already required by native E00 evaluation.

LSUI-trained V16, in-domain full test (replace both placeholders):

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN
python tools/analyze_uicf_intervention.py \
  --run-dir "experiments/<REAL_V16_LSUI_RUN>" \
  --checkpoint best_psnr \
  --data-root "<LSUI19_DATA_ROOT>" \
  --gpu 0 \
  --bootstrap-samples 5000 --bootstrap-seed 3520 \
  --viz-count 8 --viz-seed 3407
```

UIEB-trained V16, in-domain full T90 test:

```bash
python tools/analyze_uicf_intervention.py \
  --run-dir "experiments/<REAL_V16_UIEB_RUN>" \
  --checkpoint best_psnr \
  --data-root "<UIEB19_DATA_ROOT>" \
  --gpu 0 \
  --bootstrap-samples 5000 --bootstrap-seed 3520 \
  --viz-count 8 --viz-seed 3407
```

No `--dataset` is needed: the frozen run identifies it. Omit `--data-root` when
the saved root is already valid. Do not start training or call the official test
again just to use this tool. Cross-dataset `--evaluation-config` is intentionally
**not implemented**; source checkpoint and test identity must stay in-domain.

### CLI reference

| Argument | Default / meaning |
|---|---|
| `--run-dir` | required; existing canonical V16 run |
| `--checkpoint` | frozen `test.checkpoint`; selectors best_psnr/best_ssim/best_loss/last or explicit path |
| `--gpu` | native device selector, CUDA 0 when available |
| `--data-root` | saved data root, only location override |
| `--output-dir` | `RUN/result/uicf_field_intervention`; a subdirectory of it or a safe custom dedicated leaf with that name |
| `--bootstrap-samples` | 5000, >=1 |
| `--bootstrap-seed` | 3520; distinct recorded variant/metric streams |
| `--viz-count` | 8; 0 disables panels, never changes evaluated membership |
| `--viz-seed` | 3407; local RNG, selection fixed before inference |
| `--sample-ids ID1 ID2` | explicit visualization membership; overrides random count, not test membership |
| `--amp` / `--no-amp` | frozen AMP by default, always disabled on CPU; override makes old native results potentially noncomparable |
| `--num-workers` | frozen data.num_workers; >=0 |
| `--consistency-every N` | 0 = first successful sample only; N>0 also checks manifest indices divisible by N |
| `--save-all-outputs` | false; save every sample x eight variants as RGB PNG previews |
| `--overwrite` | false; safe transactional replacement of this tool's matching marked directory only |
| `--max-samples N` | debug-only even if N>=test count; no formal summary.json or paper table |

Example engineering smoke (not a paper result): append `--max-samples 2
--num-workers 0 --viz-count 1`. Do not selectively tune intervention definitions
after inspecting a checkpoint's scores.

## Outputs and failure safety

Successful complete runs publish:

```text
RUN/result/uicf_field_intervention/
  protocol.json
  per_sample_metrics.csv       # N x 8, long table of native float metrics
  paired_differences.csv       # N x 7
  diagnostics.csv              # N x 8, actual b/R/Ic/Dout diagnostics
  failed_samples.csv           # header only for success
  official_test_audit.json
  summary.json                # all eight variants, image-level paired gain CIs
  paper_table.csv              # exactly four preregistered main groups
  paper_table.md
  COMPLETED.json
  visualizations/selection.json
  visualizations/<safe_id_hash>/comparison.png
  visualizations/<safe_id_hash>/{input,gt,original,...}.png
  visualizations/<safe_id_hash>/{R_original,Ic_original}.npy
  all_outputs/<safe_id_hash>/<variant>.png   # only --save-all-outputs
```

All CSV/JSON scalar values must be finite. A mathematically infinite PSNR (exact
prediction=GT) is explicitly rejected rather than serialized as nonstandard JSON
Infinity or silently replaced; review that sample if encountered.

Every manifest sample is attempted. If any load/forward/metric/export fails, its
ID/context is recorded, the exit status is nonzero, and **no new formal result**
is published. Staging progress is retained beside the destination as
`.uicf_field_intervention.failed-<id>/`, with FAIL protocol, CSVs and
`failure_summary.json`, never successful paper tables. Ordinary exceptions and
Ctrl-C are handled; SIGKILL/power failure can leave an explicitly named staging
directory without `COMPLETED.json`. Such directories are not results.

Rerun a failed first evaluation with the same command (no overwrite necessary).
Nonempty successful destinations are refused by default. With `--overwrite`,
build a complete replacement first, then archive the previous dedicated result
to `.uicf_field_intervention.previous-<id>/` before publishing. No recursive
deletion is used; prior results remain recoverable. A failed replacement leaves
the previous successful result unchanged. Never point `--output-dir` to a run
root, result root, dataset, source, split or checkpoint directory. Symlinks and
unmarked nonempty directories are refused. Official test metrics/config/weights
are never overwritten.

Debug runs publish only `debug_summary.json` (paper_ready=false), not a normal
`summary.json` or paper table, even when every requested debug sample succeeds.
Use a separate dedicated subdirectory for debug if retaining a formal run.

## Statistics and interpretation

Compute native per-image metrics first, then equal-image arithmetic means. Gains:

```text
gain_psnr = PSNR(original) - PSNR(variant)
gain_ssim = SSIM(original) - SSIM(variant)
gain_e00  = E00(variant) - E00(original)
```

Positive always favors original. Report mean, population std (ddof=0), median,
fraction(gain>0), and paired image-level percentile 95% mean-gain CI from the
existing alignment bootstrap helper. Each variant/metric has a documented seed;
test images, not pixels or shift runs, are the independent resampling units.
These are unadjusted pointwise intervals, not multiplicity-adjusted evidence.
If a CI crosses zero, describe uncertainty. Gains that are negative/zero and
interventions that improve a fixed checkpoint are retained without filtering.

The four-row Markdown/CSV main table is generated only after full success. The
other four fixed groups remain in summary/CSVs for a supplementary table. Use
the real generated values in a paper; the synthetic test fixture's values are
**not scientific results**. A defensible conclusion is that the frozen model
depends on the original coefficient arrangement to the observed degree, not
that this proves physical correctness, transmission recovery or training
optimality. `zero_field` cannot replace independently trained w/o-UICF results.

## Validation

```bash
PYTHONPATH=. python -m pytest -q tests/test_uicf_intervention.py
PYTHONPATH=. python -m pytest -q tests/test_v16_uicf_visualization.py tests/test_uicf_alignment.py
python tools/analyze_uicf_intervention.py --help
```

Tests build nonzero tiny synthetic V16 checkpoints and disjoint RGB PNG/TSV
splits in temporary directories. They compare original per-image metrics and
saved original PNG bytes to the actual `src.v16.test` entry, test paired signs
and seeds, and inject load/consistency/metric/protocol failures. They also verify
protected-file bytes, state/RNG preservation, overwrite refusal and recoverable
replacement. CPU tests are engineering acceptance only. Real CUDA AMP equivalence
and full LSUI/UIEB checkpoint results must still be verified on AutoDL; there is
no claim that synthetic numbers measure a trained model's effectiveness.

Local validation record (Python 3.12 / Torch 2.8.0 / scikit-image 0.25.2,
CPU only): new intervention tests **35 passed**; requested visualization +
alignment regression **31 passed**; expanded regression including UICF,
Color-Query and E00 **76 passed**. In the synthetic native-test comparison,
original per-image PSNR/SSIM/E00 and saved PNG bytes matched exactly.
An attempted full `pytest -q` stopped during collection because this local
environment lacks the already-required `matplotlib` dependency for the existing
`test_chromatic_restoration_analysis.py`; the full suite is **not claimed passed**.
