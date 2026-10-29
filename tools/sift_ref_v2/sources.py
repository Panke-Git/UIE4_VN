"""Fail-closed ID pairing, immutable source audit, and legacy regression reports."""
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import sys

import cv2
import numpy as np
from PIL import Image
import yaml

from . import VERSION


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def slug(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_').lower()


def read_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    config = copy.deepcopy(config)
    p = config['protocol']
    expected = {
        'evaluation_size_wh': [256, 256],
        'sift': {'nfeatures': 0, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 1.6},
        'ratio': 0.75, 'identity_threshold_px': 3.0, 'seed': 3407, 'opencv_threads': 1,
        'raw_gt_resize': 'Pillow BILINEAR full image to 256x256',
        'grayscale': 'OpenCV RGB2GRAY on RGB uint8; no histogram equalization',
    }
    if p != expected:
        raise ValueError('Protocol differs from frozen v2/legacy protocol; parameters cannot be tuned in this tool')
    names = [m['name'] for m in config['methods']]
    if len(names) != len(set(names)) or len({slug(n) for n in names}) != len(names):
        raise ValueError('Method names/slugs must be unique')
    if 'Raw' not in names or 'ICAR-UIE' not in names:
        raise ValueError('Raw and ICAR-UIE are mandatory')
    if not config['dataset'].get('software_fixture', False):
        if not {'MARINE-Net', 'SWLGM'} <= set(names) or config['dataset']['expected_pairs'] != 428:
            raise ValueError('Production LSUI protocol requires 428 pairs and MARINE-Net/SWLGM')
        if not config.get('source_audit_dir') or not config.get('history'):
            raise ValueError('Production runs require the reviewed source audit and historical regression inputs')
    if config.get('history'):
        legacy = json.loads(Path(config['history']['summary']).read_text())
        wanted = {'ratio': 0.75, 'mutual': True, 'nfeatures': 0, 'contrast_threshold': 0.04,
                  'edge_threshold': 10.0, 'sigma': 1.6, 'threshold_px': 3.0, 'seed': 3407}
        if any(legacy['protocol']['settings'].get(k) != v for k, v in wanted.items()):
            raise ValueError('Historical SIFT settings differ from the locked protocol')
        if legacy['protocol']['geometry'] != 'identity' or legacy['protocol']['grayscale'] != expected['grayscale']:
            raise ValueError('Historical geometry/grayscale differs')
        original = yaml.safe_load(Path(config['history']['run_config']).read_text())
        if original['evaluation'] != {'resize': True, 'size': 256}:
            raise ValueError('Historical V16 evaluation is not 256x256 resize')
    config['config_path'] = str(path)
    config['config_sha256'] = digest(path)
    return config


def initialize_runtime():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(3407)
    cv2.ocl.setUseOpenCL(False)
    random.seed(3407)
    np.random.seed(3407)
    return {'version': VERSION, 'python': platform.python_version(), 'python_executable': sys.executable,
            'opencv': cv2.__version__, 'numpy': np.__version__, 'pillow': Image.__version__,
            'seed': 3407, 'opencv_threads': cv2.getNumThreads(), 'opencl': cv2.ocl.useOpenCL()}


def load_entries(config):
    manifest = Path(config['dataset']['manifest']).expanduser().resolve()
    if digest(manifest) != config['dataset']['manifest_sha256']:
        raise ValueError('Frozen manifest SHA256 mismatch')
    entries = []
    with manifest.open(encoding='utf-8', newline='') as handle:
        for row in csv.reader(handle, delimiter='\t'):
            if len(row) != 3 or not all(row):
                raise ValueError('Manifest must contain three nonempty columns, without a header')
            image_id, raw, gt = row
            if not re.fullmatch(r'[A-Za-z0-9_-]+', image_id):
                raise ValueError(f'Unsafe image ID: {image_id}')
            if Path(raw).stem != image_id or Path(gt).stem != image_id:
                raise ValueError(f'ID differs from input/reference filename: {image_id}')
            entries.append({'image_id': image_id, 'raw_relative': raw, 'gt_relative': gt})
    if len(entries) != config['dataset']['expected_pairs']:
        raise ValueError('Frozen manifest pair count mismatch')
    for key in ['image_id', 'raw_relative', 'gt_relative']:
        if len({e[key] for e in entries}) != len(entries):
            raise ValueError(f'Duplicate manifest {key}')
    root = Path(config['dataset']['data_root']).expanduser().resolve()
    for entry in entries:
        entry['paths'] = {'GT': str((root / entry['gt_relative']).resolve())}
        for method in config['methods']:
            if method['name'] == 'Raw':
                path = root / entry['raw_relative']
            else:
                name = method['filename'].format(sample_id=entry['image_id'], input_stem=Path(entry['raw_relative']).stem)
                if Path(name).name != name:
                    raise ValueError('Prediction filename template must not contain directories')
                path = Path(method['prediction_dir']).expanduser() / name
            entry['paths'][method['name']] = str(path.resolve())
    for name in ['GT'] + [m['name'] for m in config['methods']]:
        paths = [e['paths'][name] for e in entries]
        if len(set(paths)) != len(paths):
            raise ValueError(f'Duplicate paths for {name}')
    return entries


def prepare_output(output, config):
    output = Path(output).expanduser().resolve()
    forbidden = [Path(config['dataset']['data_root']).expanduser().resolve()]
    forbidden += [Path(m['prediction_dir']).expanduser().resolve() for m in config['methods'] if m['name'] != 'Raw']
    forbidden += [Path(config['history']['per_image']).parent.resolve()] if config.get('history') else []
    forbidden += [Path(config['source_audit_dir']).resolve()] if config.get('source_audit_dir') else []
    if any(output == p or output.is_relative_to(p) for p in forbidden):
        raise ValueError('Output must be isolated from inputs, history and prior audits')
    output.mkdir(parents=True, exist_ok=False)
    return output


def read_rgb(path, source=False):
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.getexif().get(274) not in (None, 1):
            raise ValueError(f'Expected RGB without EXIF rotation: {path}')
        image.load()
        if source:
            image = image.resize((256, 256), Image.Resampling.BILINEAR)
        elif image.size != (256, 256):
            raise ValueError(f'Predictions must already be 256x256: {path}')
        array = np.array(image)
    if array.dtype != np.uint8:
        raise ValueError(f'Expected uint8: {path}')
    return array


def audit_sources(config, entries, output):
    locked = {}
    locked_documents = {}
    audit_dir = config.get('source_audit_dir')
    if audit_dir:
        with (Path(audit_dir) / 'audit_sources.csv').open(encoding='utf-8-sig') as handle:
            locked = {(r['method'], r['image_id']): r for r in csv.DictReader(handle)}
        with (Path(audit_dir) / 'audit_documents.csv').open(encoding='utf-8-sig') as handle:
            locked_documents = {str(Path(r['path']).resolve()): r['sha256'] for r in csv.DictReader(handle)}
        visual = json.loads((Path(audit_dir) / 'visual_review.json').read_text())
        if visual['status'] != 'NO_OBVIOUS_FIELD_OF_VIEW_MISMATCH_IN_20_REVIEWED_SCENES':
            raise ValueError('Prior visual source alignment review is not complete')
    rows, problems = [], []
    for entry in entries:
        for method, text_path in entry['paths'].items():
            path = Path(text_path)
            row = {'method': method, 'image_id': entry['image_id'], 'path': text_path,
                   'sha256': '', 'width': '', 'height': '', 'mode': '', 'channels': '', 'dtype': '',
                   'png_bit_depth': '', 'png_color_type': '', 'prior_audit_sha256_matches': '', 'error': ''}
            try:
                row['sha256'] = digest(path)
                with Image.open(path) as image:
                    image.load()
                    row.update(width=image.width, height=image.height, mode=image.mode, channels=len(image.getbands()), dtype=str(np.asarray(image).dtype))
                    if image.mode != 'RGB' or row['dtype'] != 'uint8' or image.getexif().get(274) not in (None, 1):
                        raise ValueError('not RGB uint8 or has EXIF rotation')
                    if method not in {'Raw', 'GT'} and image.size != (256, 256):
                        raise ValueError('prediction size differs from canonical 256x256')
                    if image.format == 'PNG':
                        with path.open('rb') as handle:
                            header = handle.read(26)
                        row.update(png_bit_depth=header[24], png_color_type=header[25])
                        if header[24:26] != bytes([8, 2]):
                            raise ValueError('expected RGB PNG8 encoding')
                if locked:
                    previous = locked.get((method, entry['image_id']))
                    row['prior_audit_sha256_matches'] = bool(previous and previous['sha256'] == row['sha256'] and Path(previous['path']).resolve() == path)
                    if not row['prior_audit_sha256_matches']:
                        raise ValueError('source differs from reviewed Step 1 audit')
            except Exception as error:
                row['error'] = f'{type(error).__name__}: {error}'
                problems.append({'method': method, 'image_id': entry['image_id'], 'path': text_path, 'error': row['error']})
            rows.append(row)
    indexed = {(r['method'], r['image_id']): r for r in rows}
    for entry in entries:
        raw, gt = indexed['Raw', entry['image_id']], indexed['GT', entry['image_id']]
        if (raw['width'], raw['height']) != (gt['width'], gt['height']):
            problems.append({'method': 'Raw/GT', 'image_id': entry['image_id'], 'path': raw['path'], 'error': 'native Raw/GT dimensions differ'})
    for method in config['methods']:
        if method['name'] == 'Raw':
            continue
        folder = Path(method['prediction_dir']).expanduser().resolve()
        expected = {Path(e['paths'][method['name']]).name for e in entries}
        actual = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
        for name in sorted(actual - expected):
            problems.append({'method': method['name'], 'image_id': '', 'path': str(folder / name), 'error': 'unexpected prediction directory entry'})
    save_csv(output / 'audit_sources.csv', rows)
    save_csv(output / 'invalid_missing_sources.csv', problems, ['method', 'image_id', 'path', 'error'])
    profiles = {name: {'image_count': sum(r['method'] == name for r in rows),
                       'dimensions': dict(Counter(f"{r['width']}x{r['height']}" for r in rows if r['method'] == name))}
                for name in ['GT'] + [m['name'] for m in config['methods']]}
    document_paths = [Path(config['config_path']), Path(config['dataset']['manifest'])]
    if config.get('history'):
        document_paths += [Path(v) for v in config['history'].values()]
    if audit_dir:
        document_paths += [Path(audit_dir) / n for n in ['audit_sources.csv', 'audit_report.md', 'visual_review.json']]
    document_paths += [Path(m['provenance_file']) for m in config['methods'] if m.get('provenance_file')]
    document_paths += list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent.parent / 'sift_reference_consistency_v2.py']
    documents = []
    for path in sorted(set(document_paths)):
        try:
            record = {'path': str(path.resolve()), 'sha256': digest(path), 'bytes': path.stat().st_size}
            documents.append(record)
            previous = locked_documents.get(record['path'])
            if previous is not None and previous != record['sha256']:
                problems.append({'method': 'DOCUMENT', 'image_id': '', 'path': str(path), 'error': 'document differs from reviewed Step 1 SHA256'})
        except OSError as error:
            problems.append({'method': 'DOCUMENT', 'image_id': '', 'path': str(path), 'error': str(error)})
    save_csv(output / 'audit_documents.csv', documents, ['path', 'sha256', 'bytes'])
    save_csv(output / 'invalid_missing_sources.csv', problems, ['method', 'image_id', 'path', 'error'])
    return {'source_rows': len(rows), 'profiles': profiles, 'problems': problems, 'sources': rows, 'documents': documents}


def verify_unchanged(audit):
    changed = [r['path'] for r in audit['sources'] + audit['documents'] if digest(r['path']) != r['sha256']]
    if changed:
        raise ValueError(f'Sources changed during analysis: {changed}')
    return {'source_files_checked': len(audit['sources']), 'documents_checked': len(audit['documents']), 'changed_files': changed}


def legacy_regression(rows, config, output, selected_count):
    if not config.get('history'):
        result = {'status': 'NOT_APPLICABLE_SOFTWARE_FIXTURE', 'full_428_regression_passed': False}
        save_json(output / 'legacy_regression.json', result)
        return result
    with Path(config['history']['per_image']).open(encoding='utf-8-sig') as handle:
        history = list(csv.DictReader(handle))
    lookup = {(r['condition'], r['pair_id']): r for r in history}
    mapping = {'Raw': 'raw', 'ICAR-UIE': 'v16'}
    fields = {'keypoints_a': 'keypoints_a', 'keypoints_gt': 'keypoints_b', 'ratio_matches_a_to_gt': 'ratio_matches_a_to_b',
              'tentative': 'tentative_matches', 'verified': 'verified_matches', 'consistency_precision': 'inlier_ratio'}
    comparisons, differences = [], []
    for row in rows:
        if row['method'] not in mapping or row['K'] != 'unlimited':
            continue
        old = lookup.get((mapping[row['method']], row['image_id']))
        if old is None:
            differences.append({'method': row['method'], 'image_id': row['image_id'], 'metric': 'history_row', 'old': '', 'v2': '', 'delta': '', 'issue': 'missing historical ID'})
            continue
        for new_key, old_key in fields.items():
            original, new = float(old[old_key]), row[new_key]
            item = {'method': row['method'], 'image_id': row['image_id'], 'metric': new_key, 'old': original, 'v2': new, 'delta': new - original, 'issue': ''}
            comparisons.append(item)
            if abs(item['delta']) > (1e-12 if new_key == 'consistency_precision' else 0):
                item['issue'] = 'numeric mismatch'
                differences.append(item)
        if row['geometry_status'] != old['geometry_status']:
            differences.append({'method': row['method'], 'image_id': row['image_id'], 'metric': 'geometry_status', 'old': old['geometry_status'], 'v2': row['geometry_status'], 'delta': '', 'issue': 'status mismatch'})
    expected_rows = selected_count * 2 * len(fields)
    if len(comparisons) != expected_rows:
        raise ValueError('Regression comparison population is incomplete')
    summary = {}
    for method in mapping:
        chosen = [r for r in rows if r['method'] == method]
        summary[method] = {f'mean_{key}': float(np.mean([r[key] for r in chosen])) for key in fields}
    complete = selected_count == config['dataset']['expected_pairs']
    result = {'status': 'FAIL' if differences else ('FULL_PASS' if complete else 'SUBSET_PASS'),
              'evaluated_images': selected_count, 'expected_images': config['dataset']['expected_pairs'],
              'full_428_regression_passed': not differences and complete and selected_count == 428,
              'comparison_count': len(comparisons), 'difference_count': len(differences),
              'maximum_absolute_error': {key: max(abs(r['delta']) for r in comparisons if r['metric'] == key) for key in fields},
              'v2_means': summary, 'historical_csv_sha256': digest(config['history']['per_image']),
              'interpretation': 'Exact per-image replay on the selected IDs; subset PASS is not a full 428-image regression.'}
    save_csv(output / 'legacy_regression_per_image.csv', comparisons)
    save_csv(output / 'legacy_regression_differences.csv', differences, ['method', 'image_id', 'metric', 'old', 'v2', 'delta', 'issue'])
    save_json(output / 'legacy_regression.json', result)
    return result
