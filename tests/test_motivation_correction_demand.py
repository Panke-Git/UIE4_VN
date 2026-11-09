"""Synthetic fixtures validate code; they are never reported as research data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.v16.dataset import LSUIDataset, ManifestEntry
from tools.analyze_motivation_correction_demand import (
    DEGENERATE_ENERGY, EPSILON, compute_metrics, display_clipping, export_zip,
    load_pair, new_run_dir, parse_args, patch_means, read_manifest, residual,
    safe_pair_paths, save_figure, select_samples, summarize,
    validate_split_protocol, visualization_norms, write_json,
)


def small_metrics(a, y, p=1):
    return compute_metrics(a, y, primary_patch=p, patch_sizes=(1, 2))


def test_exact_rgb_arithmetic_and_oracle_energy():
    a = np.zeros((2, 2, 3))
    y = np.stack(([[0, .2], [.4, .6]], np.full((2, 2), .2), [[.3, .1], [.1, .3]]), axis=-1)
    np.testing.assert_array_equal(residual(a, y), y)
    m = small_metrics(a, y)
    np.testing.assert_allclose([m['mu_R'], m['mu_G'], m['mu_B']], [.3, .2, .2], atol=1e-15)
    assert m['e_raw'] == pytest.approx(.92 / 12)
    assert m['e_global'] == pytest.approx(.02)
    assert m['eta_spatial'] == pytest.approx(.02 / (.92 / 12 + EPSILON))
    assert m['spatial_std_R'] == pytest.approx(np.sqrt(.05))
    assert m['spatial_std_G'] == pytest.approx(0)
    assert m['spatial_std_B'] == pytest.approx(.1)
    assert m['channel_discrepancy'] == pytest.approx(.1)
    assert m['channel_variance'] == pytest.approx(1 / 450)
    assert m['patch_variance_1'] == pytest.approx(.02)
    assert m['patch_variance_2'] == pytest.approx(0)
    assert m['mean_abs_R'] == pytest.approx(.3)
    assert m['mu_R_minus_G'] == pytest.approx(.1)


def test_signed_residual_and_absolute_means_do_not_cancel():
    a = np.full((2, 2, 3), .5)
    y = a.copy()
    y[..., 0] += [[-.2, .2], [-.2, .2]]
    y[..., 1] += .1
    y[..., 2] -= .3
    m = small_metrics(a, y)
    assert m['mu_R'] == pytest.approx(0, abs=1e-15)
    assert m['mu_G'] == pytest.approx(.1)
    assert m['mu_B'] == pytest.approx(-.3)
    assert m['mean_abs_R'] == pytest.approx(.2)
    assert m['mean_abs_B'] == pytest.approx(.3)


def test_zero_residual_is_degenerate_and_json_finite(tmp_path):
    a = np.zeros((2, 2, 3))
    m = small_metrics(a, a)
    assert m['degenerate'] and m['eta_spatial'] is None
    assert m['e_raw'] == m['e_global'] == 0
    assert m['patch_spatial_retention_1'] is None
    write_json(tmp_path / 'zero.json', m)
    assert 'NaN' not in (tmp_path / 'zero.json').read_text()
    assert 'Infinity' not in (tmp_path / 'zero.json').read_text()


def test_near_zero_energy_is_not_epsilon_dominated():
    a = np.zeros((2, 2, 3))
    y = np.full_like(a, np.sqrt(DEGENERATE_ENERGY) / 2)
    m = small_metrics(a, y)
    assert m['degenerate'] and m['eta_spatial'] is None


def test_constant_color_offset_fully_explained():
    a = np.full((2, 2, 3), .3)
    m = small_metrics(a, a + [.1, -.2, .3])
    assert m['e_global'] < 1e-30
    assert m['eta_spatial'] < 1e-28
    assert m['channel_discrepancy'] == pytest.approx(.5)


def test_global_offset_minimizes_squared_error():
    rng = np.random.default_rng(3520)
    a, y = rng.random((32, 32, 3)), rng.random((32, 32, 3))
    d = residual(a, y)
    mu = d.mean(axis=(0, 1))
    m = compute_metrics(a, y)
    for perturbation in ([.01, 0, 0], [0, -.1, .1], [.3, .3, .3]):
        assert np.mean((d - mu - perturbation) ** 2) > m['e_global']
    assert m['e_global'] <= m['e_raw'] + 1e-12
    assert m['e_raw'] - m['e_global'] == pytest.approx(np.mean(mu**2), abs=1e-14)


def test_patch_averaging_known_values_and_population_variance():
    d = np.arange(4 * 4 * 3).reshape(4, 4, 3) / 100
    expected = np.array([[d[:2, :2].mean(axis=(0, 1)), d[:2, 2:].mean(axis=(0, 1))],
                         [d[2:, :2].mean(axis=(0, 1)), d[2:, 2:].mean(axis=(0, 1))]])
    np.testing.assert_allclose(patch_means(d, 2), expected)
    m = compute_metrics(np.zeros_like(d), d, primary_patch=2, patch_sizes=(1, 2, 4))
    assert m['patch_variance_2'] == pytest.approx(np.var(expected, axis=(0, 1), ddof=0).mean())
    assert m['patch_variance_4'] == pytest.approx(0)


def test_high_frequency_only_is_not_misread_as_coarse_structure():
    yy, xx = np.indices((32, 32))
    d = np.repeat((((xx + yy) % 2 * 2 - 1) * .1)[..., None], 3, axis=-1)
    m = compute_metrics(np.full_like(d, .5), .5 + d)
    assert m['eta_spatial'] == pytest.approx(1)
    assert m['patch_variance_16'] == pytest.approx(0, abs=1e-28)


def test_nested_patch_variance_bounds():
    rng = np.random.default_rng(3520)
    m = compute_metrics(rng.random((256, 256, 3)), rng.random((256, 256, 3)))
    assert 0 <= m['patch_variance_32'] <= m['patch_variance_16'] <= m['patch_variance_8'] <= m['e_global']


@pytest.mark.parametrize('size', [0, -1, 3])
def test_patch_size_invalid_or_not_divisible(size):
    with pytest.raises(ValueError):
        patch_means(np.zeros((4, 4, 3)), size)


@pytest.mark.parametrize('bad', [np.full((2, 2, 3), np.nan), np.full((2, 2, 3), np.inf),
                               np.full((2, 2, 3), -1), np.full((2, 2, 3), 1.01)])
def test_invalid_numeric_domain_rejected(bad):
    with pytest.raises(ValueError):
        residual(np.zeros((2, 2, 3)), bad)


def test_bad_array_shape_rejected():
    with pytest.raises(ValueError, match='HWC RGB'):
        residual(np.zeros((2, 2, 3)), np.zeros((2, 2, 4)))


def make_pair(tmp_path, *, a_mode='RGB', b_size=(19, 23), corrupt=False):
    root = tmp_path / 'images'
    for sub in ('input', 'gt'):
        (root / sub).mkdir(parents=True)
    e = ManifestEntry('sample', 'input/sample.png', 'gt/sample.png')
    Image.new(a_mode, (19, 23), 42 if a_mode == 'L' else (11, 89, 201)).save(root / e.input_relative)
    Image.new('RGB', b_size, (41, 79, 211)).save(root / e.gt_relative)
    if corrupt:
        (root / e.input_relative).write_bytes(b'not an image')
    return root, e


@pytest.mark.parametrize('header', ['', 'sample_id\tinput_path\tgt_path\n', 'id\tinput\tgt\n'])
def test_three_column_manifest_compatibility(tmp_path, header):
    p = tmp_path / 'test.tsv'
    p.write_text(header + '3\tVal/input/3.jpg\tVal/GT/3.jpg\n')
    assert read_manifest(p) == [ManifestEntry('3', 'Val/input/3.jpg', 'Val/GT/3.jpg')]


@pytest.mark.parametrize('text,pattern', [('', 'empty'), ('1\tx.png\n', 'three non-empty'),
                                       ('1\t\tx.png\n', 'three non-empty'),
                                       ('1\t/absolute.png\tx.png\n', 'relative')])
def test_manifest_invalid_format(tmp_path, text, pattern):
    p = tmp_path / 'bad.tsv'; p.write_text(text)
    with pytest.raises(ValueError, match=pattern):
        read_manifest(p)


def split_fixtures(tmp_path):
    manifests = {}
    for name, sid in [('train', 'a'), ('validation', 'b'), ('test', 'c')]:
        p = tmp_path / (name + '.tsv')
        p.write_text(f'{sid}\tinput/{sid}.png\tgt/{sid}.png\n')
        manifests[name] = p
    return manifests


def test_duplicate_sample_id_detected_even_when_paths_differ(tmp_path):
    manifests = split_fixtures(tmp_path)
    manifests['test'].write_text('c\tinput/c.png\tgt/c.png\nc\tinput/d.png\tgt/d.png\n')
    with pytest.raises(ValueError, match='Duplicate sample_id'):
        validate_split_protocol(manifests, {'train': 1, 'validation': 1, 'test': 2})


def test_cross_split_leakage_detected(tmp_path):
    manifests = split_fixtures(tmp_path)
    manifests['test'].write_text(manifests['train'].read_text())
    with pytest.raises(ValueError, match='Cross-split leakage'):
        validate_split_protocol(manifests, dict.fromkeys(manifests, 1))


def test_wrong_split_count_detected(tmp_path):
    manifests = split_fixtures(tmp_path)
    with pytest.raises(ValueError, match='test count is 1, expected 2'):
        validate_split_protocol(manifests, {'train': 1, 'validation': 1, 'test': 2})


def test_missing_file_has_clear_context(tmp_path):
    root, e = make_pair(tmp_path)
    (root / e.gt_relative).unlink()
    with pytest.raises(RuntimeError, match='missing GT.*sample_id=sample'):
        load_pair(e, root, {'resize': True, 'size': 256})


def test_mismatched_original_dimensions_rejected_before_resize(tmp_path):
    root, e = make_pair(tmp_path, b_size=(20, 23))
    with pytest.raises(RuntimeError, match='size mismatch'):
        load_pair(e, root, {'resize': True, 'size': 256})


def test_non_rgb_never_silently_converted(tmp_path):
    root, e = make_pair(tmp_path, a_mode='L')
    with pytest.raises(RuntimeError, match='non-RGB'):
        load_pair(e, root, {'resize': True, 'size': 256})


def test_corrupt_image_logged_as_unreadable(tmp_path):
    root, e = make_pair(tmp_path, corrupt=True)
    with pytest.raises(RuntimeError, match='unreadable'):
        load_pair(e, root, {'resize': True, 'size': 256})


def test_wrong_data_root_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match='data.root is unavailable'):
        load_pair(ManifestEntry('a', 'a.png', 'a.png'), tmp_path / 'missing', {'resize': True, 'size': 256})


def test_manifest_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match='escapes data.root'):
        safe_pair_paths(ManifestEntry('a', '../a.png', '../b.png'), tmp_path)


def test_pil_rgb_bilinear_float32_matches_existing_dataset(tmp_path):
    root, e = make_pair(tmp_path)
    # A nonconstant RGB fixture detects interpolation, layout and RGB swapping.
    rng = np.random.default_rng(3520)
    Image.fromarray(rng.integers(0, 256, (23, 19, 3), dtype=np.uint8)).save(root / e.input_relative)
    manifest = tmp_path / 'test.tsv'
    manifest.write_text(f'{e.sample_id}\t{e.input_relative}\t{e.gt_relative}\n')
    evaluation = {'resize': True, 'size': 256}
    existing = LSUIDataset(manifest, root, 'test', 256, {'hflip': True, 'rot90': True}, True, 'reflect', evaluation)
    batch = existing[0]
    a, y, meta = load_pair(e, root, evaluation)
    np.testing.assert_array_equal(a, batch['input'].numpy().transpose(1, 2, 0))
    np.testing.assert_array_equal(y, batch['target'].numpy().transpose(1, 2, 0))
    assert a.dtype == y.dtype == np.float32
    assert meta['original_dimensions_wh'] == [19, 23]
    assert a.shape == (256, 256, 3)
    np.testing.assert_array_equal(load_pair(e, root, evaluation)[0], a)


def test_oracle_errors_stay_unclipped():
    a = np.zeros((2, 2, 3)); a[:1, :, 0] = 1
    y = a.copy(); y[..., 0] = 1
    m = small_metrics(a, y)
    mu = residual(a, y).mean(axis=(0, 1))
    clipped_error = np.mean((y - np.clip(a + mu, 0, 1)) ** 2)
    assert m['e_global'] == pytest.approx(1 / 12)
    assert m['e_global'] > clipped_error
    assert m['oracle_scalar_clipped_fraction'] == pytest.approx(1 / 6)
    assert m['oracle_pixel_clipped_fraction'] == pytest.approx(.5)


def test_display_saturation_does_not_modify_raw_data_or_statistics():
    a = np.zeros((2, 2, 3)); y = np.ones_like(a)
    before = y.copy(); m1 = small_metrics(a, y)
    d = residual(a, y)
    clipped = display_clipping(d, d - d.mean(axis=(0, 1)), .01, .01)
    assert clipped['display_signed_clipped_fraction'] == 1
    np.testing.assert_array_equal(y, before)
    assert small_metrics(a, y) == m1


def test_common_signed_norm_and_zero_center():
    signed, magnitude = visualization_norms()
    assert (signed.vmin, signed.vcenter, signed.vmax) == (-.5, 0, .5)
    assert signed(0) == .5
    assert (magnitude.vmin, magnitude.vmax) == (0, 1)
    for channel in 'RGB':
        assert float(signed(.2)) == pytest.approx(.7)
    full, full_magnitude = visualization_norms(True)
    assert (full.vmin, full.vcenter, full.vmax) == (-1, 0, 1)
    assert full_magnitude.vmax == pytest.approx(np.sqrt(3))


def selection_rows(n=10):
    return [{'dataset': 'lsui', 'sample_id': str(i), 'input_path': f'/i/{i}.png', 'gt_path': f'/g/{i}.png',
             'input_relative': f'i/{i}.png', 'gt_relative': f'g/{i}.png',
             'original_width': 42, 'original_height': 32, 'analysis_width': 256, 'analysis_height': 256,
             'degenerate': False, 'patch_spatial_variance': i * .001, 'channel_discrepancy': i * .02}
            for i in range(n)]


def test_selection_deterministic_order_independent_and_nonextreme():
    rows = selection_rows()
    selection = select_samples(rows)
    assert selection == select_samples(rows[::-1])
    assert [s['sample_id'] for s in selection] == ['5', '7', '6']
    assert all(s['sample_id'] != '9' for s in selection)


def test_selection_ties_use_lexicographic_id_and_degenerate_exclusion():
    rows = selection_rows(6)
    for r in rows:
        r['patch_spatial_variance'] = r['channel_discrepancy'] = .01
    rows[2]['degenerate'] = True
    chosen = select_samples(rows)
    assert chosen == select_samples(rows[::-1])
    assert len({s['sample_id'] for s in chosen}) == 3
    assert not any(s['sample_id'] == '2' for s in chosen)


def test_equal_image_summary_and_null_counts():
    s = summarize([0, 2, None])
    assert s == {'mean': 1., 'median': 1., 'std': 1., 'p05': .1, 'p95': 1.9, 'valid_count': 2, 'null_count': 1}
    assert summarize([None])['mean'] is None
    with pytest.raises(ArithmeticError, match='Non-finite'):
        summarize([np.nan])


def test_outputs_never_overwrite_even_same_timestamp(tmp_path):
    first = new_run_dir(tmp_path, 'lsui', stamp='fixed')
    write_json(first / 'keep.json', {'keep': True})
    second = new_run_dir(tmp_path, 'lsui', stamp='fixed')
    assert first != second
    assert json.loads((first / 'keep.json').read_text()) == {'keep': True}
    with pytest.raises(FileExistsError):
        write_json(first / 'keep.json', {'keep': False})
    archive = export_zip(first)
    assert archive.is_file()
    with pytest.raises(FileExistsError):
        export_zip(first)


def test_figure_output_refuses_overwrite(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.plot([0, 1], [1, 0])
    save_figure(fig, tmp_path / 'figure', formats=('png',))
    with pytest.raises(FileExistsError):
        save_figure(fig, tmp_path / 'figure', formats=('png',))
    plt.close(fig)


def test_cli_test_only_and_smoke_marker():
    args = parse_args(['--dataset', 'lsui', '--data-root', '/unavailable', '--sample-limit', '4'])
    assert args.split == 'test' and args.sample_limit == 4 and args.seed == 3520
    with pytest.raises(SystemExit):
        parse_args(['--dataset', 'lsui', '--data-root', '/unavailable', '--split', 'train'])
    with pytest.raises(SystemExit):
        parse_args(['--dataset', 'uieb', '--data-root', '/unavailable', '--sample-limit', '0'])


def test_actual_panels_share_norm_and_preserve_raw_values(tmp_path, monkeypatch):
    import tools.analyze_motivation_correction_demand as tool
    d = np.zeros((8, 8, 3)); d[..., 0] = .8; d[..., 1] = -.2
    sample = {'input': np.full_like(d, .2), 'target': np.full_like(d, .2),
              'delta': d, 'patch_size': 8, 'label': 'synthetic fixture'}
    captured = []
    def inspect(fig, path, formats):
        images = [ax.images[0] for ax in fig.axes if ax.images]
        signed = [im for im in images if im.get_cmap().name == 'coolwarm']
        assert len(signed) == 6
        assert len({id(im.norm) for im in signed}) == 1
        assert signed[0].norm(0) == .5
        np.testing.assert_array_equal(signed[0].get_array(), d[..., 0])
        captured.append(True)
        return []
    monkeypatch.setattr(tool, 'save_figure', inspect)
    tool.render_grid([sample, sample], tmp_path / 'unused', formats=())
    assert captured == [True]


def test_cli_failed_root_persists_failed_status_without_fallback(tmp_path, capsys):
    from tools.analyze_motivation_correction_demand import main
    code = main(['--dataset', 'lsui', '--data-root', str(tmp_path / 'absent'),
                 '--output-dir', str(tmp_path / 'outputs'), '--no-figures'])
    assert code == 1
    run = next((tmp_path / 'outputs').glob('run_*'))
    fatal = json.loads((run / 'metadata/fatal_error.json').read_text())
    assert fatal['analysis_status'] == 'FAILED'
    assert 'data.root is unavailable' in fatal['reason']
    assert not (run / 'statistics/dataset_summary.json').exists()
    assert 'ANALYSIS_STATUS: FAILED' in capsys.readouterr().out


@pytest.mark.parametrize('full_range', [False, True])
@pytest.mark.parametrize('kind', ['main', 'patch'])
def test_adjacent_colorbar_endpoint_labels_do_not_overlap(tmp_path, monkeypatch, full_range, kind):
    import tools.analyze_motivation_correction_demand as tool
    d = np.full((32, 32, 3), .1)
    sample = {'input': np.full_like(d, .2), 'target': np.full_like(d, .3),
              'delta': d, 'patch_size': 8, 'label': 'synthetic fixture'}
    def inspect(fig, path, formats):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        colorbars = [ax for ax in fig.axes if not ax.images]
        assert len(colorbars) == 2
        last_magnitude = colorbars[0].get_xticklabels()[-1].get_window_extent(renderer)
        first_signed = colorbars[1].get_xticklabels()[0].get_window_extent(renderer)
        assert last_magnitude.x1 + 2 < first_signed.x0
        return []
    monkeypatch.setattr(tool, 'save_figure', inspect)
    tool.render_grid([sample], tmp_path / 'unused', kind=kind, full_range=full_range, formats=())


@pytest.mark.parametrize('n', [1, 3])
@pytest.mark.parametrize('kind', ['main', 'oracle', 'patch'])
def test_grid_titles_and_footer_have_safe_physical_margins(tmp_path, monkeypatch, n, kind):
    import tools.analyze_motivation_correction_demand as tool
    d = np.full((32, 32, 3), .1)
    sample = {'input': np.full_like(d, .2), 'target': np.full_like(d, .3),
              'delta': d, 'patch_size': 8, 'label': 'synthetic fixture'}
    def inspect(fig, path, formats):
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        footer = fig.texts[-1].get_window_extent(renderer)
        canvas = fig.get_window_extent(renderer)
        for ax in fig.axes:
            if ax.title.get_text():
                title = ax.title.get_window_extent(renderer)
                assert title.y1 + 2 < canvas.y1
                assert title.x0 >= canvas.x0 and title.x1 <= canvas.x1
            if not ax.images:
                labels = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_visible()]
                assert footer.y1 + 2 < min(b.y0 for b in labels)
        assert footer.y0 >= 0
        return []
    monkeypatch.setattr(tool, 'save_figure', inspect)
    tool.render_grid([sample]*n, tmp_path / 'unused', kind=kind, formats=())
