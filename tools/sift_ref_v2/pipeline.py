"""Staged regression gate, common-reference experiments, validation and reports."""
import csv
from pathlib import Path
import statistics as standard_statistics

import numpy as np

from .features import Features, detect, match, metrics, retain
from .sources import (audit_sources, digest, initialize_runtime, legacy_regression, load_entries,
                      prepare_output, read_config, read_rgb, save_csv, save_json, slug, verify_unchanged)
from .statistics import bootstrap, paired_rows, paired_summary, summarize


def feature_path(output, method, image_id):
    return output / 'features' / slug(method) / f'{image_id}.npz'


def correspondence_path(output, method, image_id, budget):
    return output / 'correspondences' / slug(method) / f'{image_id}_K{budget}.npz'


def measure(output, entry, method, features, reference, budget):
    selected = retain(features, budget)
    matches = match(selected, reference)
    matches.save(correspondence_path(output, method, entry['image_id'], budget))
    row = {'image_id': entry['image_id'], 'method': method, 'K': budget,
           **metrics(selected, reference, matches, len(features), budget)}
    if budget == 'unlimited':
        row.update(tentative_matches=row['tentative'], identity_consistent_matches_at_3px=row['verified'])
    else:
        row.update(tentative_at_K=row['tentative'], verified_at_K=row['verified'],
                   reference_coverage_at_K=row['reference_coverage'], verified_per_retained_at_K=row['verified_per_retained'])
    return row


def compare_all(rows, methods, budgets):
    pairs, summaries = [], []
    contrasts = [(m['name'], 'Raw') for m in methods if m['name'] != 'Raw']
    contrasts += [('ICAR-UIE', m['name']) for m in methods if m.get('sota', False)]
    for budget in budgets:
        for method, baseline in contrasts:
            chosen = paired_rows(rows, method, baseline, budget)
            pairs.extend(chosen)
            summaries.append(paired_summary(chosen))
    return pairs, summaries


def independent_validation(output, config, entries, budgets, audit):
    """Re-read actual CSVs, aggregate using stdlib, and inspect saved coordinate masks."""
    checks = {'csv_aggregate_checks': 0, 'coordinate_checks': 0, 'budget_checks': 0}
    for stem in ['baseline_unlimited', 'fixed_budget']:
        with (output / f'{stem}_per_image.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        with (output / f'{stem}_summary.csv').open() as handle:
            summaries = list(csv.DictReader(handle))
        for summary in summaries:
            selected = [r for r in rows if r['method'] == summary['method'] and str(r['K']) == str(summary['K'])]
            if len(selected) != len(entries) or int(summary['image_count']) != len(selected):
                raise ValueError('CSV summary population mismatch')
            for metric in ['detected_total', 'retained', 'keypoints_gt', 'tentative', 'verified', 'reference_coverage', 'consistency_precision', 'verified_per_retained']:
                values = [float(r[metric]) for r in selected]
                for label, expected in [('mean', standard_statistics.mean(values)), ('median', standard_statistics.median(values))]:
                    if not np.isclose(float(summary[f'{label}_{metric}']), expected, rtol=0, atol=1e-10):
                        raise ValueError(f'CSV aggregation mismatch: {metric}')
                    checks['csv_aggregate_checks'] += 1
            if int(summary['budget_shortage_count']) != sum(int(r['budget_shortage']) for r in selected):
                raise ValueError('Budget shortage summary mismatch')
        for row in rows:
            detected, retained = int(row['detected_total']), int(row['retained'])
            expected = detected if row['K'] == 'unlimited' else min(int(row['K']), detected)
            if retained != expected or int(row['budget_shortage']) != int(row['K'] != 'unlimited' and detected < int(row['K'])):
                raise ValueError('Budget handling inconsistent with CSV')
            if not 0 <= int(row['verified']) <= int(row['tentative']) <= retained:
                raise ValueError('Impossible CSV correspondence count')
            checks['budget_checks'] += 1
    requested = [e for e in entries if e['image_id'] in {'1930', '220', '895', '1757'}]
    other = [e for e in entries if e not in requested]
    rng = np.random.default_rng(3407)
    sample = requested + [other[int(i)] for i in rng.choice(len(other), size=min(16, len(other)), replace=False)]
    for entry in sample:
        gt = Features.load(feature_path(output, 'GT', entry['image_id']))
        for method in config['methods']:
            source = Features.load(feature_path(output, method['name'], entry['image_id']))
            for budget in budgets:
                with np.load(correspondence_path(output, method['name'], entry['image_id'], budget), allow_pickle=False) as record:
                    q, t, mask = record['query_indices'], record['reference_indices'], record['consistent']
                    distance_squared = ((source.points[q].astype(np.float64) - gt.points[t].astype(np.float64)) ** 2).sum(axis=1)
                    if not np.array_equal(mask, distance_squared <= 9.0):
                        raise ValueError('Saved mask disagrees with independently computed Identity@3px')
                    if len(set(q.tolist())) != len(q) or len(set(t.tolist())) != len(t):
                        raise ValueError('Saved correspondences are not one-to-one')
                    if not np.array_equal(record['identity_matrix'], np.eye(3)):
                        raise ValueError('Unexpected non-identity coordinate transform')
                    if not set(q.tolist()) <= set(retain(source, budget).indices.tolist()):
                        raise ValueError('Match outside retained feature budget')
                    checks['coordinate_checks'] += 1
    checks.update(matching_review_ids=[e['image_id'] for e in sample], all_checks_passed=True,
                  source_immutability=verify_unchanged(audit))
    save_json(output / 'validation.json', checks)
    return checks


def run(config_path, output_path, only_ids=None, budgets=None, dry_run=False, regression_only=False):
    config = read_config(config_path)
    runtime = initialize_runtime()
    if runtime['opencv_threads'] != 1 and not config['dataset'].get('software_fixture', False):
        raise RuntimeError('This OpenCV build did not honor setNumThreads(1); use the audited WSL duo_yolo environment')
    all_entries = load_entries(config)
    ids = only_ids or []
    if len(ids) != len(set(ids)) or set(ids) - {e['image_id'] for e in all_entries}:
        raise ValueError('Duplicate/unknown --only-ids')
    entries = [e for e in all_entries if not ids or e['image_id'] in set(ids)]
    budgets = budgets or ['unlimited', 128, 256, 512]
    if any(k != 'unlimited' and (not isinstance(k, int) or k < 1) for k in budgets):
        raise ValueError('K must be unlimited or a positive integer')
    budgets = ['unlimited'] + sorted({k for k in budgets if k != 'unlimited'})
    output = prepare_output(output_path, config)
    scope = {'manifest_images': len(all_entries), 'evaluated_images': len(entries),
             'partial': len(entries) != len(all_entries), 'image_ids': [e['image_id'] for e in entries],
             'dry_run': dry_run, 'regression_only': regression_only}
    resolved = {**config, 'runtime': runtime, 'scope': scope, 'effective_K': budgets,
                'output_dir': str(output), 'gt_reference': 'One complete detectAndCompute library per ID, shared across every method and K',
                'tie_break': ['-response', 'y', 'x', 'size', 'angle', 'octave', 'class_id', 'original_index'],
                'empty_denominator': 'zero, with zero cases separately reported',
                'bootstrap': {'replicates': 5000, 'seed': 3407, 'unit': 'image', 'confidence': 0.95},
                'interpretation': 'reference-geometrically consistent correspondences; not ground-truth matching accuracy'}
    save_json(output / 'config_resolved.json', resolved)
    print(f'Runtime: {runtime}', flush=True)
    audit = audit_sources(config, all_entries, output)
    save_json(output / 'audit_summary.json', {k: v for k, v in audit.items() if k not in {'sources', 'documents'}})
    print(f"Audit: {len(all_entries)} IDs; {audit['source_rows']} sources; {len(audit['problems'])} problems; selected {len(entries)} IDs", flush=True)
    if audit['problems']:
        (output / 'report.md').write_text('Source audit failed. No SIFT computation was performed. See invalid_missing_sources.csv.\n')
        raise ValueError(f'Source audit failed; see {output / "invalid_missing_sources.csv"}')
    if dry_run:
        save_json(output / 'run_status.json', {'stage': 'DRY_RUN_AUDIT_PASSED', 'complete': False, 'scope': scope})
        (output / 'report.md').write_text('Dry run: source IDs, dimensions, encodings, provenance and SHA256 passed. No feature extraction or experiments were run.\n')
        return {'output_dir': str(output), 'stage': 'DRY_RUN_AUDIT_PASSED'}
    try:
        # Phase 1: no SOTA evaluation or fixed-budget measurements before the regression gate.
        baseline = []
        for number, entry in enumerate(entries, 1):
            gt = detect(read_rgb(entry['paths']['GT'], source=True))
            gt.save(feature_path(output, 'GT', entry['image_id']))
            for name in ['Raw', 'ICAR-UIE']:
                feature = detect(read_rgb(entry['paths'][name], source=name == 'Raw'))
                feature.save(feature_path(output, name, entry['image_id']))
                baseline.append(measure(output, entry, name, feature, gt, 'unlimited'))
            if number % 10 == 0 or number == len(entries):
                print(f'Regression {number}/{len(entries)}', flush=True)
        regression = legacy_regression(baseline, config, output, len(entries))
        print(f"Legacy regression: {regression['status']}", flush=True)
        if regression['status'] == 'FAIL':
            save_csv(output / 'baseline_unlimited_per_image.csv', baseline)
            (output / 'report.md').write_text('Historical unlimited regression FAILED. New v2 and historical experiments are not declared directly comparable. No SOTA or fixed-budget experiment was run. Inspect legacy_regression_differences.csv; compare runtime, preprocessing and input hashes recorded here.\n')
            raise ValueError('Legacy regression failed; no SOTA/fixed-budget results were generated')
        if regression_only:
            save_csv(output / 'baseline_unlimited_per_image.csv', baseline)
            save_csv(output / 'baseline_unlimited_summary.csv', summarize(baseline))
            save_json(output / 'run_status.json', {'stage': 'REGRESSION_ONLY', 'complete': False, 'scope': scope})
            verify_unchanged(audit)
            return {'output_dir': str(output), 'stage': regression['status']}
        fixed = []
        for number, entry in enumerate(entries, 1):
            gt = Features.load(feature_path(output, 'GT', entry['image_id']))
            for method in config['methods']:
                name = method['name']
                cached = feature_path(output, name, entry['image_id'])
                if name in {'Raw', 'ICAR-UIE'}:
                    feature = Features.load(cached)
                else:
                    feature = detect(read_rgb(entry['paths'][name]))
                    feature.save(cached)
                    baseline.append(measure(output, entry, name, feature, gt, 'unlimited'))
                for budget in budgets:
                    if budget != 'unlimited':
                        fixed.append(measure(output, entry, name, feature, gt, budget))
            if number % 10 == 0 or number == len(entries):
                print(f'Experiments {number}/{len(entries)}; K={budgets}', flush=True)
        # Stable canonical row order; no timing/path fields in metric CSVs.
        for rows in [baseline, fixed]:
            rows.sort(key=lambda r: (r['method'], str(r['K']), r['image_id']))
        base_summary, fixed_summary = summarize(baseline), summarize(fixed)
        save_csv(output / 'baseline_unlimited_per_image.csv', baseline)
        save_csv(output / 'baseline_unlimited_summary.csv', base_summary)
        save_csv(output / 'fixed_budget_per_image.csv', fixed, list(fixed[0]) if fixed else list(baseline[0])[:-2] + ['tentative_at_K', 'verified_at_K', 'reference_coverage_at_K', 'verified_per_retained_at_K'])
        save_csv(output / 'fixed_budget_summary.csv', fixed_summary, list(base_summary[0]))
        all_rows = baseline + fixed
        pair_rows, pair_summaries = compare_all(all_rows, config['methods'], budgets)
        save_csv(output / 'paired_differences_per_image.csv', pair_rows)
        save_csv(output / 'paired_differences_summary.csv', pair_summaries)
        # Select strongest on the predeclared main metric, never on PSNR or favorable examples.
        sota_names = [m['name'] for m in config['methods'] if m.get('sota', False)]
        scores = {r['method']: r['mean_verified'] for r in fixed_summary if str(r['K']) == '256' and r['method'] in sota_names}
        strongest = sorted([name for name, score in scores.items() if score == max(scores.values())]) if scores else []
        ci = []
        if 256 in budgets:
            for summary in pair_summaries:
                if str(summary['K']) == '256':
                    contrast = paired_rows(all_rows, summary['method'], summary['baseline'], 256)
                    ci.append({**bootstrap(contrast), 'baseline_is_strongest_included_sota': int(summary['baseline'] in strongest)})
        save_csv(output / 'paired_bootstrap_ci.csv', ci, list(ci[0]) if ci else ['method', 'baseline', 'K', 'metric', 'ci95_low', 'ci95_high'])
        save_json(output / 'strongest_sota.json', {'metric': 'macro mean verified@256', 'scores': scores, 'tied_strongest': strongest,
                                                   'figure_representative': strongest[0] if strongest else None,
                                                   'tie_rule': 'Report all exact ties; alphabetic representative for the figure only',
                                                   'selection_caveat': 'Selection on this evaluation set; CIs describe fixed observed contrasts, without selection/multiplicity adjustment.'})
        validation = independent_validation(output, config, entries, budgets, audit)
        # Scientific figures only after complete statistics, with all 428 images present.
        figure = {'generated': False, 'reason': 'Requires full 428-image run, K=256 and an eligible SOTA'}
        if not scope['partial'] and len(entries) == 428 and strongest and 256 in budgets:
            from .figures import qualitative
            figure = qualitative(output, entries, strongest[0], fixed)
        from .report import write_report
        write_report(output, resolved, base_summary, fixed_summary, pair_summaries, ci, regression, strongest, figure, all_rows)
        save_json(output / 'run_status.json', {'stage': 'COMPLETE' if not scope['partial'] else 'SMOKE_COMPLETE', 'complete': not scope['partial'],
                                             'scope': scope, 'legacy_regression': regression['status'], 'validation': validation, 'figure': figure})
        return {'output_dir': str(output), 'stage': 'COMPLETE' if not scope['partial'] else 'SMOKE_COMPLETE',
                'legacy_regression': regression['status'], 'strongest_sota': strongest}
    except Exception as error:
        save_json(output / 'run_status.json', {'stage': 'FAILED', 'complete': False, 'scope': scope,
                                             'error': f'{type(error).__name__}: {error}'})
        raise
