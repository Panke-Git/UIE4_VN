"""Synthetic protocol checks; external-data regression runs explicitly on WSL."""
import copy
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from tools.sift_ref_v2.features import (Features, detect, identity_mask, match, metrics, mutual_pairs,
                       ratio_pairs, response_order, retain, validate_one_to_one)
from tools.sift_ref_v2.pipeline import run
from tools.sift_ref_v2.sources import digest, initialize_runtime, legacy_regression
from tools.sift_ref_v2.statistics import bootstrap, paired_rows, summarize


def texture():
    rng = np.random.default_rng(3407)
    image = cv2.GaussianBlur(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), (3, 3), .5)
    for _ in range(40):
        x, y = rng.integers(5, 250, size=2)
        cv2.circle(image, (int(x), int(y)), 3, (240, 60, 150), -1)
    return image


def artificial(n):
    p = np.zeros((n, 7), dtype=np.float64)
    p[:, 0] = np.arange(n)
    p[:, 4] = np.arange(n) % 3
    descriptors = np.zeros((n, 128), dtype=np.float32)
    descriptors[:, 0] = np.arange(n)
    return Features(p, descriptors, np.arange(n))


def test_identity_and_inclusive_euclidean_threshold():
    initialize_runtime()
    f = detect(texture())
    result = match(f, f)
    assert result.consistent.all() and len(result.consistent) > 50
    np.testing.assert_array_equal(result.query_indices, result.reference_indices)
    assert identity_mask([[0, 0]] * 4, [[0, 0], [3, 0], [1.8, 2.4], [3.0001, 0]]).tolist() == [True, True, True, False]
    assert identity_mask([[0, 0]], [[2.2, 2.2]]).tolist() == [False]


@pytest.mark.parametrize('n,m', [(0, 0), (0, 3), (1, 3), (3, 1), (1, 1)])
def test_too_few_descriptor_neighbors(n, m):
    result = match(artificial(n), artificial(m))
    assert len(result.consistent) == 0
    assert ratio_pairs(None, np.empty((0, 128), np.float32)) == []


def test_response_sorting_preserves_descriptor_indices_and_deterministic_ties():
    f = artificial(9)
    f.keypoints[:, 1] = [8, 3, 9, 1, 0, 9, 7, 6, 2]
    order = response_order(f)
    expected = sorted(range(9), key=lambda i: (-f.keypoints[i, 4], f.keypoints[i, 1], f.keypoints[i, 0], f.keypoints[i, 2], f.keypoints[i, 3], f.keypoints[i, 5], f.keypoints[i, 6], i))
    assert order.tolist() == expected
    chosen = retain(f, 4)
    np.testing.assert_array_equal(chosen.descriptors[:, 0], chosen.indices)
    assert chosen.indices.tolist() == expected[:4]
    assert retain(f, 'unlimited') is f


def test_shortage_46_and_zero_features_remain_in_macro_statistics():
    rows = []
    for image_id, n in [('1757', 46), ('blank', 0)]:
        all_features = artificial(n)
        selected = retain(all_features, 256)
        row = {'image_id': image_id, 'method': 'Raw', 'K': 256,
               **metrics(selected, artificial(90), match(selected, artificial(90)), n, 256)}
        rows.append(row)
    summary = summarize(rows)[0]
    assert rows[0]['retained'] == 46 and rows[0]['budget_shortage'] == 1
    assert summary['image_count'] == 2 and summary['mean_retained'] == 23
    assert summary['budget_shortage_fraction'] == 1.0 and summary['zero_detected_count'] == 1


def test_mutual_ratio_intersection_is_not_crosscheck_or_duplicate_gt_counting():
    assert mutual_pairs([(0, 0, 1.)], []) == []
    assert mutual_pairs([(0, 0, 1.), (1, 0, .5)], [(0, 1, .5)]) == [(1, 0, .5)]
    with pytest.raises(ValueError, match='one-to-one'):
        validate_one_to_one([0, 1], [2, 2])


def test_feature_io_and_same_method_repeated_detection_are_identical(tmp_path):
    initialize_runtime()
    a, b = detect(texture()), detect(texture())
    for name in ['keypoints', 'descriptors', 'indices']:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    a.save(tmp_path / 'a.npz')
    restored = Features.load(tmp_path / 'a.npz')
    np.testing.assert_array_equal(restored.descriptors, a.descriptors)


def test_unlimited_matches_independent_legacy_implementation():
    # Test-only import. The v2 runtime does not import any v1 modules.
    from tools.sift_eval.matching import Settings, match_images
    initialize_runtime()
    raw = texture()
    enhanced = np.clip(raw.astype(float) * .9 + 15, 0, 255).astype(np.uint8)
    gt = detect(raw)
    for image in [raw, enhanced]:
        source = detect(image)
        result = match(retain(source, 'unlimited'), gt)
        new = metrics(source, gt, result, len(source), 'unlimited')
        legacy = match_images(Image.fromarray(image), Image.fromarray(raw), 'identity', Settings()).metrics
        assert new['keypoints_a'] == legacy['keypoints_a']
        assert new['keypoints_gt'] == legacy['keypoints_b']
        assert new['tentative'] == legacy['tentative_matches']
        assert new['verified'] == legacy['verified_matches']
        assert new['consistency_precision'] == legacy['inlier_ratio']


@pytest.fixture
def fixture_config(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    for role in ['input', 'GT', 'ICAR', 'SOTA']:
        (data / role).mkdir()
        for image_id, array in [('1930', texture()), ('1757', np.zeros((256, 256, 3), np.uint8))]:
            Image.fromarray(array).save(data / role / f'{image_id}.png')
    manifest = tmp_path / 'test.tsv'
    manifest.write_text('1930\tinput/1930.png\tGT/1930.png\n1757\tinput/1757.png\tGT/1757.png\n')
    config = json.loads((Path(__file__).resolve().parents[2] / 'configs/sift/reference_v2.home_wsl.json').read_text())
    config['dataset'] = {'name': 'Synthetic', 'expected_pairs': 2, 'software_fixture': True,
                         'manifest': str(manifest), 'manifest_sha256': digest(manifest), 'data_root': str(data)}
    config.pop('source_audit_dir')
    config.pop('history')
    config['methods'] = [{'name': 'Raw', 'sota': False},
                         {'name': 'ICAR-UIE', 'sota': False, 'filename': '{sample_id}.png', 'prediction_dir': str(data / 'ICAR')},
                         {'name': 'MockSOTA', 'sota': True, 'filename': '{sample_id}.png', 'prediction_dir': str(data / 'SOTA')}]
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return path


def test_end_to_end_gt_cache_summary_and_repeated_metric_files(fixture_config, tmp_path, monkeypatch):
    import tools.sift_ref_v2.pipeline as pipeline
    original, counts = pipeline.detect, []
    def counted(image):
        counts.append(1)
        return original(image)
    monkeypatch.setattr(pipeline, 'detect', counted)
    outputs = [tmp_path / 'first', tmp_path / 'second']
    for output in outputs:
        run(fixture_config, output, budgets=[128, 256, 512])
        assert len(list((output / 'features/gt').glob('*.npz'))) == 2
        check = json.loads((output / 'validation.json').read_text())
        assert check['all_checks_passed'] and check['coordinate_checks'] == 24
        with (output / 'fixed_budget_per_image.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2 * 3 * 3
        assert len({r['keypoints_gt'] for r in rows if r['image_id'] == '1930'}) == 1
    assert len(counts) == 2 * 2 * 4  # GT extracted once per image, three source methods once each.
    for name in ['baseline_unlimited_per_image.csv', 'baseline_unlimited_summary.csv',
                 'fixed_budget_per_image.csv', 'fixed_budget_summary.csv', 'paired_bootstrap_ci.csv',
                 'paired_differences_per_image.csv', 'paired_differences_summary.csv']:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_dry_run_extracts_no_features_and_invalid_inputs_are_not_skipped(fixture_config, tmp_path, monkeypatch):
    monkeypatch.setattr('tools.sift_ref_v2.pipeline.detect', lambda _: pytest.fail('dry run extracted SIFT'))
    run(fixture_config, tmp_path / 'dry', dry_run=True)
    assert not (tmp_path / 'dry/features').exists()
    (tmp_path / 'data/SOTA/1930.png').unlink()
    with pytest.raises(ValueError, match='audit failed'):
        run(fixture_config, tmp_path / 'missing', dry_run=True)
    assert 'FileNotFoundError' in (tmp_path / 'missing/invalid_missing_sources.csv').read_text()
    assert not (tmp_path / 'missing/features').exists()
    with pytest.raises(FileExistsError):
        run(fixture_config, tmp_path / 'dry', dry_run=True)


def test_bootstrap_is_image_paired_deterministic_and_handles_zero_baseline():
    rows = [{'image_id': str(i), 'method': method, 'K': 256, 'verified': i + offset,
             'tentative': i + offset, 'reference_coverage': 0., 'consistency_precision': 0., 'retained': 20}
            for method, offset in [('Raw', 0), ('ICAR-UIE', 5)] for i in range(4)]
    pairs = paired_rows(rows, 'ICAR-UIE', 'Raw', 256)
    first = bootstrap(pairs)
    assert first == bootstrap(pairs)
    assert first['mean_verified_delta'] == first['ci95_low'] == first['ci95_high'] == 5
    assert pairs[0]['verified_relative_change'] is None


def test_regression_reports_individual_differences_without_changing_history(tmp_path):
    history = tmp_path / 'history.csv'
    fields = ['condition', 'pair_id', 'keypoints_a', 'keypoints_b', 'ratio_matches_a_to_b', 'tentative_matches', 'verified_matches', 'inlier_ratio', 'geometry_status']
    with history.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in ['raw', 'v16']:
            writer.writerow(dict(zip(fields, [condition, '1757', 46, 300, 40, 39, 39, 1., 'ok'])))
    before = history.read_bytes()
    rows = [{'method': name, 'image_id': '1757', 'K': 'unlimited', 'keypoints_a': 46, 'keypoints_gt': 300,
             'ratio_matches_a_to_gt': 40, 'tentative': 39, 'verified': 39, 'consistency_precision': 1., 'geometry_status': 'ok'} for name in ['Raw', 'ICAR-UIE']]
    config = {'history': {'per_image': str(history)}, 'dataset': {'expected_pairs': 1}}
    assert legacy_regression(rows, config, tmp_path, 1)['status'] == 'FULL_PASS'
    rows[0]['verified'] = 38
    result = legacy_regression(rows, config, tmp_path, 1)
    assert result['status'] == 'FAIL' and result['difference_count'] == 1
    assert history.read_bytes() == before


def test_full_428_raw_icar_legacy_regression_artifact():
    # Run after actual WSL full analysis: SIFT_V2_RESULT_DIR=... python -m pytest ...
    directory = os.environ.get('SIFT_V2_RESULT_DIR')
    if not directory:
        pytest.skip('Full 428-image external-data regression is exercised on WSL, not synthetic fixtures')
    output = Path(directory)
    result = json.loads((output / 'legacy_regression.json').read_text())
    assert result['status'] == 'FULL_PASS' and result['full_428_regression_passed']
    assert result['difference_count'] == 0
    assert result['v2_means']['Raw']['mean_keypoints_a'] == pytest.approx(534.3668224299065, abs=1e-12)
    assert result['v2_means']['Raw']['mean_verified'] == pytest.approx(332.6588785046729, abs=1e-12)
    assert result['v2_means']['ICAR-UIE']['mean_keypoints_a'] == pytest.approx(770.2570093457944, abs=1e-12)
    assert result['v2_means']['ICAR-UIE']['mean_verified'] == pytest.approx(484.0607476635514, abs=1e-12)
