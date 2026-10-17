#!/usr/bin/env python3
"""Read-only preflight check for an unpacked Haze4K train/test/haze/gt/trans dataset.

Usage: python haze4k_preflight.py --data-root /absolute/path/to/Haze4K [--hash-gt]
This tool never modifies the dataset or splits it. Requires Pillow for size checks.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def images(path: Path) -> list[Path]:
    return sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXT), key=lambda p: p.name) if path.is_dir() else []

def digest(p: Path) -> str:
    sha = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description='Haze4K read-only preflight')
    ap.add_argument('--data-root', required=True, type=Path)
    ap.add_argument('--hash-gt', action='store_true', help='Check identical GT files across official train/test (slower).')
    args = ap.parse_args()
    root = args.data_root.expanduser().resolve()
    print(f'Absolute data root: {root}')
    try:
        from PIL import Image
    except ImportError:
        print('ERROR: Pillow is needed for image shape/mode checks: pip install Pillow', file=sys.stderr)
        return 2
    if not root.is_dir():
        print('ERROR: data root does not exist', file=sys.stderr)
        return 2
    any_error = False
    gt_by_split = {}
    for split in ('train', 'test'):
        sub = root / split
        haze_dir, gt_dir, trans_dir = sub / 'haze', sub / 'gt', sub / 'trans'
        for req in (haze_dir, gt_dir):
            if not req.is_dir():
                print(f'ERROR: missing {req}', file=sys.stderr)
                any_error = True
        if any_error and (not haze_dir.is_dir() or not gt_dir.is_dir()):
            continue
        haze, gt, trans = images(haze_dir), images(gt_dir), images(trans_dir)
        ignored = sorted(p.name for p in haze_dir.iterdir() if p.is_file() and p.suffix.lower() not in EXT)
        if ignored:
            print(f'[{split}] ignored non-image files in haze/: {ignored[:20]}')
        gt_index = defaultdict(list)
        for p in gt:
            gt_index[p.stem].append(p)
        problems = []
        per_clean = Counter()
        sampled = []
        gt_sizes = {}
        for h in haze:
            if '_' not in h.stem:
                problems.append(f'{h.name}: no underscore in haze stem')
                continue
            clean_id = h.stem.split('_', 1)[0]
            choices = gt_index.get(clean_id, [])
            if len(choices) != 1:
                problems.append(f'{h.name}: clean ID={clean_id!r} has {len(choices)} GT matches')
                continue
            g = choices[0]
            per_clean[clean_id] += 1
            if len(sampled) < 3:
                sampled.append(f'{h.name}  ->  {g.name}')
            try:
                with Image.open(h) as himg:
                    hs = himg.size
                    himg.load()  # Force full file decode rather than only reading the header.
                if g not in gt_sizes:
                    with Image.open(g) as gimg:
                        gt_sizes[g] = gimg.size
                        gimg.load()  # Validate every used ground-truth image.
                if hs != gt_sizes[g]:
                    problems.append(f'{h.name}: size {hs} != GT {g.name} {gt_sizes[g]}')
            except Exception as e:
                problems.append(f'{h.name} or {g.name}: unreadable ({e})')
        unused = [p.name for p in gt if per_clean[p.stem] == 0]
        multiple_gt_stems = {x: len(ps) for x, ps in gt_index.items() if len(ps) > 1}
        if unused:
            problems.append(f'{len(unused)} unused GT images, examples: {unused[:5]}')
        if multiple_gt_stems:
            problems.append(f'duplicate GT identifiers with different suffixes: {list(multiple_gt_stems.items())[:5]}')
        print(f'\n[{split}] haze={len(haze)}, gt={len(gt)}, trans={len(trans)} (trans ignored for training)')
        print(f' paired hazy={sum(per_clean.values())}, unique clean IDs={len(per_clean)}')
        print(f' haze samples per clean ID: {dict(sorted(Counter(per_clean.values()).items()))}')
        print(' examples:')
        for s in sampled:
            print(f'   {s}')
        if problems:
            any_error = True
            print(f' ERROR count={len(problems)}; first 10:')
            for p in problems[:10]:
                print('   ' + p)
        else:
            print(' Pairing and image-size inspection: PASS')
        if split == 'train' and len(haze) != 3000:
            print(' WARNING: expected commonly 3000 train haze images; verify this release.')
        if split == 'test' and len(haze) != 1000:
            print(' WARNING: expected commonly 1000 test haze images; verify this release.')
        gt_by_split[split] = gt
    if args.hash_gt and all(s in gt_by_split for s in ('train', 'test')):
        train_digests = defaultdict(list)
        for p in gt_by_split['train']:
            train_digests[digest(p)].append(p)
        overlaps = [(p, train_digests[digest(p)]) for p in gt_by_split['test'] if digest(p) in train_digests]
        if overlaps:
            any_error = True
            print(f'\nERROR: {len(overlaps)} test GT files are byte-identical to train GT. Examples:')
            for p, matches in overlaps[:5]:
                print(f'  {p} == {matches[0]}')
        else:
            print('\nGT train/test exact-byte duplicate check: PASS')
    print('\nResult:', 'FAIL (investigate before training)' if any_error else 'PASS (file pairing, decoding and size checks; split to be generated by Codex implementation)')
    return 1 if any_error else 0

if __name__ == '__main__':
    raise SystemExit(main())
