"""Publication-sized source-backed figures, generated only after complete statistics."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .features import Features, retain
from .sources import read_rgb, save_json


def qualitative(output, entries, strongest, rows):
    from .pipeline import correspondence_path, feature_path

    ids, methods = ['1930', '895'], ['Raw', strongest, 'ICAR-UIE']
    entries = {e['image_id']: e for e in entries}
    selected = {(r['method'], r['image_id']): r for r in rows if r['K'] == 256}
    if not all(i in entries for i in ids):
        return {'generated': False, 'reason': 'Predeclared qualitative IDs are absent'}
    records = []
    with plt.rc_context({'svg.fonttype': 'none', 'svg.hashsalt': '3407', 'font.family': 'DejaVu Sans'}):
        fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.95))
        fig.subplots_adjust(left=.055, right=.99, top=.91, bottom=.13, wspace=.06, hspace=.25)
        for row_index, image_id in enumerate(ids):
            for col, method in enumerate(methods):
                ax = axes[row_index, col]
                rgb = read_rgb(entries[image_id]['paths'][method], source=method == 'Raw')
                full = Features.load(feature_path(output, method, image_id))
                kept = retain(full, 256)
                with np.load(correspondence_path(output, method, image_id, 256), allow_pickle=False) as matches:
                    indices = set(matches['query_indices'][matches['consistent']].tolist())
                mask = np.asarray([i in indices for i in kept.indices])
                score = selected[method, image_id]['verified']
                if int(mask.sum()) != score:
                    raise ValueError('Figure markers do not match CSV verified count')
                ax.imshow(rgb, interpolation='none', origin='upper')
                for verified, color in [(False, '#ff3db8'), (True, '#00ef9c')]:
                    points = kept.points[mask == verified]
                    ax.scatter(points[:, 0], points[:, 1], s=3.0, facecolors='none', edgecolors=color,
                               linewidths=.32, zorder=3 if verified else 2)
                ax.set(xlim=(-.5, 255.5), ylim=(255.5, -.5), xticks=[], yticks=[])
                for border in ax.spines.values():
                    border.set_linewidth(.45)
                if row_index == 0:
                    ax.set_title(method, fontsize=7, pad=4)
                if col == 0:
                    ax.set_ylabel(f'ID {image_id}', fontsize=6.5, labelpad=3)
                ax.set_xlabel(f'Verified@256 = {score}', fontsize=6, labelpad=3)
                records.append({'image_id': image_id, 'method': method, 'K': 256, 'retained': len(kept),
                                'verified': score, 'plotted_consistent': int(mask.sum()),
                                'plotted_unverified': int((~mask).sum())})
        handles = [Line2D([], [], linestyle='', marker='o', markersize=3, markerfacecolor='none', markeredgewidth=.6,
                          color=color, label=text) for color, text in [('#00cf86', 'Geometry consistent'), ('#e42a9a', 'Unverified')]]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .015), ncol=2,
                   frameon=False, fontsize=6, handletextpad=.3, columnspacing=1)
        fig.savefig(output / 'qualitative_2x3.png', dpi=300, metadata={'Software': 'sift-reference-consistency-v2'})
        fig.savefig(output / 'qualitative_2x3.svg', dpi=300, metadata={'Date': None, 'Creator': 'sift-reference-consistency-v2'})
        plt.close(fig)
    result = {'generated': True, 'ids': ids, 'methods': methods, 'dpi': 300, 'width_inches': 3.5,
              'selection': 'IDs 1930 and 895 were specified before v2 statistics; method selected by full-set mean verified@256.',
              'drawing': 'All retained K=256 keypoints shown; same marker size and threshold for every method; no correspondence lines.',
              'svg': 'Raster RGB backgrounds; editable vector text, frames and keypoint markers.', 'panels': records}
    save_json(output / 'qualitative_metadata.json', result)
    return result
