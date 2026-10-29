"""SIFT feature budgets and reciprocal ratio matching, without network dependencies."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Features:
    # x, y, size, angle, response, octave, class_id; original detector order.
    keypoints: np.ndarray
    descriptors: np.ndarray
    indices: np.ndarray

    def __post_init__(self):
        if self.keypoints.shape != (len(self.indices), 7) or self.descriptors.shape != (len(self.indices), 128):
            raise ValueError("Keypoints, descriptors and original indices must remain aligned")

    def __len__(self):
        return len(self.indices)

    @property
    def points(self):
        return self.keypoints[:, :2].astype(np.float32)

    def take(self, indices):
        return Features(self.keypoints[indices], np.ascontiguousarray(self.descriptors[indices]), self.indices[indices])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keypoints=self.keypoints, descriptors=self.descriptors, indices=self.indices)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['keypoints'], data['descriptors'], data['indices'])


def detect(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB uint8; no implicit grayscale/contrast conversion")
    detector = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    points, descriptors = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    values = np.asarray([[*p.pt, p.size, p.angle, p.response, p.octave, p.class_id] for p in points], dtype=np.float64).reshape(-1, 7)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    return Features(values, np.ascontiguousarray(descriptors, dtype=np.float32), np.arange(len(points), dtype=np.int64))


def response_order(features):
    """Descending response; deterministic spatial/attribute ties, then original index."""
    p = features.keypoints
    return np.lexsort((features.indices, p[:, 6], p[:, 5], p[:, 3], p[:, 2], p[:, 0], p[:, 1], -p[:, 4]))


def retain(features, budget):
    # Preserve original OpenCV order for exact unlimited regression, including tie behavior in BF.
    if budget == 'unlimited':
        return features
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("Budget must be a positive integer or unlimited")
    return features.take(response_order(features)[:min(budget, len(features))])


def ratio_pairs(a, b):
    if a is None or b is None or len(a) == 0 or len(b) < 2:
        return []
    neighbors = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(a, b, k=2)
    return [(m[0].queryIdx, m[0].trainIdx, m[0].distance) for m in neighbors
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]


def mutual_pairs(forward, reverse):
    reciprocal = {(train, query) for query, train, _ in reverse}
    selected = sorted((m for m in forward if (m[0], m[1]) in reciprocal), key=lambda m: (m[0], m[1]))
    validate_one_to_one([m[0] for m in selected], [m[1] for m in selected])
    return selected


def validate_one_to_one(query_indices, reference_indices):
    if len(query_indices) != len(set(query_indices)) or len(reference_indices) != len(set(reference_indices)):
        raise ValueError("Correspondences violate the one-to-one keypoint constraint")


def identity_mask(a, b, threshold=3.0):
    a, b = np.asarray(a, np.float32).reshape(-1, 2), np.asarray(b, np.float32).reshape(-1, 2)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Identity verification requires paired finite coordinates")
    return np.linalg.norm(a - b, axis=1) <= threshold


@dataclass
class Correspondences:
    query_indices: np.ndarray
    reference_indices: np.ndarray
    consistent: np.ndarray
    distances: np.ndarray
    forward_count: int

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, query_indices=self.query_indices, reference_indices=self.reference_indices,
                            consistent=self.consistent, descriptor_distances=self.distances,
                            identity_matrix=np.eye(3))


def match(source, reference):
    forward = ratio_pairs(source.descriptors, reference.descriptors)
    reverse = ratio_pairs(reference.descriptors, source.descriptors)
    pairs = mutual_pairs(forward, reverse)
    query = np.asarray([m[0] for m in pairs], np.int64)
    target = np.asarray([m[1] for m in pairs], np.int64)
    mask = identity_mask(source.points[query], reference.points[target])
    return Correspondences(source.indices[query], reference.indices[target], mask,
                           np.asarray([m[2] for m in pairs], np.float32), len(forward))


def metrics(source, reference, result, detected_total, budget):
    tentative, verified = len(result.consistent), int(result.consistent.sum())
    unique = len(np.unique(result.reference_indices[result.consistent]))
    validate_one_to_one(result.query_indices.tolist(), result.reference_indices.tolist())
    if unique != verified or verified > len(source):
        raise ValueError("Verified counts must count unique reference keypoints once")
    status = 'no_keypoints' if not len(source) or not len(reference) else ('ok' if tentative else 'no_matches')
    return {
        'detected_total': detected_total, 'retained': len(source), 'keypoints_a': len(source),
        'keypoints_gt': len(reference), 'budget_shortage': int(budget != 'unlimited' and detected_total < budget),
        'ratio_matches_a_to_gt': result.forward_count, 'tentative': tentative, 'verified': verified,
        'unique_verified_gt_keypoints': unique, 'consistency_precision': verified / tentative if tentative else 0.0,
        'reference_coverage': unique / len(reference) if len(reference) else 0.0,
        'verified_per_retained': verified / len(source) if len(source) else 0.0,
        'geometry_status': status,
    }
