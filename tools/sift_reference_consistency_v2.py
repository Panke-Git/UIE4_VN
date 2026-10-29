#!/usr/bin/env python3
"""Saved-image SIFT reference consistency. Never imports a network or trains/infers."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sift_ref_v2.pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path, help='Must not exist; no overwrite mode')
    parser.add_argument('--dry-run', action='store_true', help='Decode/hash all configured sources; no SIFT computation')
    parser.add_argument('--only-ids', nargs='+', help='Exact image IDs, space/comma separated; subset always labelled partial')
    parser.add_argument('--K', nargs='+', default=['unlimited', '128', '256', '512'], help='Feature budgets; unlimited is always included for regression')
    parser.add_argument('--regression-only', action='store_true', help='Only replay Raw/ICAR unlimited history, before any SOTA/fixed-budget experiment')
    args = parser.parse_args(argv)
    budgets = [k if k == 'unlimited' else int(k) for k in args.K]
    ids = [i for token in args.only_ids or [] for i in token.split(',') if i]
    result = run(args.config, args.output_dir, ids, budgets, args.dry_run, args.regression_only)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
