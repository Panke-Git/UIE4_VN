"""Image-level macro summaries and seeded paired bootstrap; zeros are retained."""
import numpy as np

METRICS = ['detected_total', 'retained', 'keypoints_a', 'keypoints_gt', 'ratio_matches_a_to_gt',
           'tentative', 'verified', 'unique_verified_gt_keypoints', 'consistency_precision',
           'reference_coverage', 'verified_per_retained']


def relative(delta, baseline):
    return float(delta / baseline) if baseline else None


def summarize(rows):
    result = []
    keys = sorted({(r['method'], str(r['K'])) for r in rows})
    for method, budget in keys:
        subset = [r for r in rows if r['method'] == method and str(r['K']) == budget]
        if len({r['image_id'] for r in subset}) != len(subset):
            raise ValueError("Duplicate image IDs in summary")
        row = {'method': method, 'K': budget, 'image_count': len(subset),
               'budget_shortage_count': sum(r['budget_shortage'] for r in subset),
               'budget_shortage_fraction': sum(r['budget_shortage'] for r in subset) / len(subset),
               'zero_detected_count': sum(r['detected_total'] == 0 for r in subset),
               'zero_gt_keypoints_count': sum(r['keypoints_gt'] == 0 for r in subset),
               'zero_tentative_count': sum(r['tentative'] == 0 for r in subset),
               'zero_verified_count': sum(r['verified'] == 0 for r in subset)}
        for metric in METRICS:
            values = [r[metric] for r in subset]
            row.update({f'mean_{metric}': float(np.mean(values)), f'median_{metric}': float(np.median(values))})
        result.append(row)
    return result


def paired_rows(rows, method, baseline, budget):
    selected = [r for r in rows if str(r['K']) == str(budget)]
    left = {r['image_id']: r for r in selected if r['method'] == method}
    right = {r['image_id']: r for r in selected if r['method'] == baseline}
    if not left or left.keys() != right.keys():
        raise ValueError(f"Paired population differs: {method} vs {baseline}, K={budget}")
    result = []
    for image_id in sorted(left):
        row = {'image_id': image_id, 'method': method, 'baseline': baseline, 'K': budget}
        for metric in ['verified', 'tentative', 'reference_coverage', 'consistency_precision', 'retained']:
            a, b = left[image_id][metric], right[image_id][metric]
            row.update({f'{metric}_method': a, f'{metric}_baseline': b,
                        f'{metric}_delta': a - b, f'{metric}_relative_change': relative(a - b, b)})
        result.append(row)
    return result


def paired_summary(pairs):
    deltas = np.asarray([r['verified_delta'] for r in pairs])
    baseline_mean = float(np.mean([r['verified_baseline'] for r in pairs]))
    return {'method': pairs[0]['method'], 'baseline': pairs[0]['baseline'], 'K': pairs[0]['K'],
            'image_count': len(pairs), 'mean_verified_method': float(np.mean([r['verified_method'] for r in pairs])),
            'mean_verified_baseline': baseline_mean, 'mean_verified_delta': float(deltas.mean()),
            'median_verified_delta': float(np.median(deltas)),
            'verified_relative_change_of_means': relative(float(deltas.mean()), baseline_mean),
            'increased': int((deltas > 0).sum()), 'equal': int((deltas == 0).sum()), 'decreased': int((deltas < 0).sum()),
            'zero_verified_baseline_count': sum(r['verified_baseline'] == 0 for r in pairs)}


def bootstrap(pairs, replicates=5000, seed=3407):
    # One image index is drawn together for both methods. Identical draws across contrasts.
    delta = np.asarray([r['verified_delta'] for r in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(replicates, len(delta)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method='linear')
    return {**paired_summary(pairs), 'metric': 'verified@256', 'bootstrap_unit': 'image',
            'bootstrap_replicates': replicates, 'seed': seed, 'ci95_low': float(low), 'ci95_high': float(high),
            'ci_method': 'paired percentile; numpy quantile linear'}
