#!/usr/bin/env python3
"""Independent audit: reopen originals and recompute without importing analyzer.

Uses explicit block loops and second moments for variance, as a separate route
from the analyzer's reshape/centered-variance implementation. Read-only except
for the explicitly named new audit JSON (exclusive creation).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    source = json.loads((run / 'metadata/source_manifest.json').read_text())
    summary = json.loads((run / 'statistics/dataset_summary.json').read_text())
    protocol = json.loads((run / 'metadata/protocol.json').read_text())
    with (run / 'statistics/per_image_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    with Path(source['manifests']['test']['path']).open() as handle:
        manifest = list(csv.reader(handle, delimiter='\t'))
    assert summary['analysis_status'] == 'PASS' and not summary['is_smoke_test']
    assert len(rows) == len(manifest) == summary['expected_pairs'] == summary['valid_pairs']
    assert summary['failed_pairs'] == 0
    assert [r['sample_id'] for r in rows] == [r[0] for r in manifest]
    assert protocol['variance_ddof'] == 0 and protocol['epsilon'] == 1e-12
    independent = []
    max_error = 0.
    numeric_comparisons = 0
    for index, (row, entry) in enumerate(zip(rows, manifest)):
        images = []
        for p in (entry[1], entry[2]):
            path = Path(source['data_root']) / p
            with Image.open(path) as im:
                assert im.mode == 'RGB'
                assert list(im.size) == [int(row['original_width']), int(row['original_height'])]
                encoded = np.asarray(im.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255.)
                images.append(encoded.astype(np.float64))
        a, y = images
        d = y - a
        n = d.shape[0] * d.shape[1]
        mu = d.sum(axis=(0, 1)) / n
        second = np.square(d).sum(axis=(0, 1)) / n
        spatial_variances = np.maximum(second - mu * mu, 0)
        eraw = float(second.sum() / 3)
        remaining = y - (a + mu)
        eglobal = float(np.square(remaining).sum() / (3*n))
        oracle_outside = np.logical_or(a + mu < 0, a + mu > 1)
        m = {'residual_magnitude_mean': float(np.sqrt(np.square(d).sum(axis=2)).sum() / n),
             'e_raw': eraw, 'e_global': eglobal,
             'eta_spatial': None if eraw <= 1e-10 else eglobal/(eraw+1e-12),
             'channel_discrepancy': float(mu.max()-mu.min()),
             'channel_variance': float(np.sum((mu-mu.sum()/3)**2)/3),
             'oracle_scalar_clipped_fraction': float(oracle_outside.sum()/(3*n)),
             'oracle_pixel_clipped_fraction': float(oracle_outside.any(axis=2).sum()/n)}
        for c, label in enumerate('RGB'):
            m['mu_'+label] = float(mu[c])
            m['mean_abs_'+label] = float(np.abs(d[...,c]).sum()/n)
            m['spatial_std_'+label] = float(np.sqrt(spatial_variances[c]))
            m[f'display_signed_{label}_clipped_fraction'] = float((np.abs(d[...,c])>.5).sum()/n)
        for first, second_c in [('R','G'),('R','B'),('G','B')]:
            m[f'mu_{first}_minus_{second_c}'] = m['mu_'+first]-m['mu_'+second_c]
        m['display_signed_clipped_fraction'] = float((np.abs(d)>.5).sum()/(3*n))
        m['display_magnitude_clipped_fraction'] = float((np.sqrt(np.square(d).sum(axis=2))>1).sum()/n)
        m['display_remaining_clipped_fraction'] = float((np.sqrt(np.square(remaining).sum(axis=2))>1).sum()/n)
        for size in sorted(set(protocol['sensitivity_patch_sizes'])):
            means = np.array([np.sum(d[t:t+size,l:l+size],axis=(0,1))/(size*size)
                              for t in range(0,256,size) for l in range(0,256,size)])
            centered = means - mu
            var = float(np.square(centered).sum()/means.size)
            m[f'patch_variance_{size}'] = var
            m[f'patch_spatial_retention_{size}'] = None if eglobal<=1e-10 else var/eglobal
        m['patch_spatial_variance'] = m[f"patch_variance_{protocol['patch_size']}"]
        for key, value in m.items():
            if value is None:
                assert row[key] == '', (row['sample_id'],key,row[key])
            else:
                err = abs(float(row[key])-value)
                max_error = max(max_error,err)
                assert err < 2e-10, (row['sample_id'],key,row[key],value,err)
            numeric_comparisons += 1
        assert (row['degenerate']=='True') == (eraw<=1e-10)
        assert eglobal<=eraw+1e-12
        assert abs(eglobal-float(spatial_variances.mean()))<1e-12
        info = source['images'][index]
        assert digest(info['input_path'])==info['input_sha256']
        assert digest(info['gt_path'])==info['gt_sha256']
        independent.append(m)
    summary_comparisons=0
    for key in independent[0]:
        values=np.array([r[key] for r in independent if r[key] is not None])
        if not len(values):
            continue
        expected={'mean':float(values.sum()/len(values)), 'median':float(np.quantile(values,.5)),
                  'std':float(np.sqrt(np.sum((values-values.sum()/len(values))**2)/len(values))),
                  'p05':float(np.quantile(values,.05,method='linear')), 'p95':float(np.quantile(values,.95,method='linear'))}
        for stat, value in expected.items():
            assert abs(value-summary['metrics'][key][stat])<2e-10,(key,stat,value,summary['metrics'][key][stat])
            summary_comparisons+=1
        assert summary['metrics'][key]['valid_count']==len(values)
    for split, info in source['manifests'].items():
        assert digest(info['path'])==info['sha256'],split
    for rel, h in source['protected_hashes'].items():
        assert digest(Path(source['repository_root'])/rel)==h,rel
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source['repository_root'],text=True).strip()==source['repository_sha']
    selected=json.loads((run/'metadata/selected_samples.json').read_text())
    used=set()
    for item in selected:
        key=item['selection_metric']; q=item['target_quantile']
        indices=[i for i,r in enumerate(rows) if r['degenerate']=='False']
        indices.sort(key=lambda i:(independent[i][key],rows[i]['sample_id'],rows[i]['input_path'],rows[i]['gt_path']))
        rank=int(np.floor(q*(len(indices)-1)+.5))
        candidates=sorted(range(len(indices)),key=lambda j:(abs(j-rank),j))
        chosen=next(indices[j] for j in candidates if rows[indices[j]]['sample_id'] not in used)
        assert rows[chosen]['sample_id']==item['sample_id']
        used.add(item['sample_id'])
    result={'status':'PASS','dataset':summary['dataset'],'verified_pairs':len(rows),
            'independent_per_image_numeric_comparisons':numeric_comparisons,
            'independent_summary_comparisons':summary_comparisons,
            'max_per_image_absolute_difference':max_error,
            'all_original_image_hashes_match':True,'source_and_manifest_hashes_match':True,
            'selection_reproduced_from_independent_metrics':True,
            'method':'independent original-image reopen; explicit patch loops; variance by second moment',
            'run_dir':str(run),'verifier_sha256':digest(Path(__file__)),
            'threshold_counts':{key:{'count':sum(r[key]>t for r in independent if r[key] is not None),'threshold':t}
                                for key,t in [('eta_spatial',.5),('patch_spatial_retention_16',.5),('channel_discrepancy',.05)]}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
