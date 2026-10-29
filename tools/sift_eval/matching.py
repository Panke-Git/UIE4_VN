"""CPU SIFT matching. Geometric consistency is not ground-truth match accuracy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Settings:
    ratio: float = 0.75
    mutual: bool = True
    nfeatures: int = 0
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    threshold_px: float = 3.0
    confidence: float = 0.999
    max_iters: int = 5000
    seed: int = 3407

    def __post_init__(self):
        if not 0 < self.ratio < 1:
            raise ValueError("ratio must be in (0,1)")
        if self.nfeatures < 0 or self.max_iters < 1:
            raise ValueError("nfeatures must be >= 0 and max_iters >= 1")
        for key in ("contrast_threshold", "edge_threshold", "sigma", "threshold_px"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be in (0,1)")
        if not 0 <= self.seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")


@dataclass
class MatchResult:
    metrics: dict
    points_a: np.ndarray
    points_b: np.ndarray
    distances: np.ndarray
    inliers: np.ndarray
    matrix: np.ndarray | None

    def save(self, path):
        np.savez_compressed(
            path, points_a=self.points_a, points_b=self.points_b,
            descriptor_distances=self.distances, inliers=self.inliers,
            geometry_matrix=self.matrix if self.matrix is not None else np.empty((0, 0)),
        )


def verify_geometry(a, b, geometry, settings):
    """Identity uses known pixel alignment; other models are independently estimated."""
    if geometry not in {"identity", "homography", "fundamental"}:
        raise ValueError(f"Unknown geometry: {geometry}")
    a, b = np.asarray(a, np.float32).reshape(-1, 2), np.asarray(b, np.float32).reshape(-1, 2)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correspondences must be finite matching Nx2 arrays")
    mask = np.zeros(len(a), dtype=bool)
    if geometry == "identity":
        mask = np.linalg.norm(a - b, axis=1) <= settings.threshold_px
        return mask, np.eye(3), "ok" if len(a) else "no_matches"
    minimum = 4 if geometry == "homography" else 8
    if len(a) < minimum:
        return mask, None, "insufficient_matches"
    cv2.setRNGSeed(settings.seed)
    try:
        if geometry == "homography":
            matrix, estimated = cv2.findHomography(
                a, b, cv2.RANSAC, settings.threshold_px,
                maxIters=settings.max_iters, confidence=settings.confidence,
            )
        else:
            matrix, estimated = cv2.findFundamentalMat(
                a, b, cv2.FM_RANSAC, settings.threshold_px,
                settings.confidence, settings.max_iters,
            )
    except cv2.error:
        return mask, None, "estimation_failed"
    rank = 3 if geometry == "homography" else 2
    if (matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) < rank or estimated is None
            or estimated.size != len(a) or np.count_nonzero(estimated) < minimum):
        return mask, None, "estimation_failed"
    return estimated.ravel().astype(bool), matrix, "ok"


def _ratio_matches(descriptors_a, descriptors_b, ratio):
    if descriptors_a is None or descriptors_b is None or len(descriptors_b) < 2:
        return []
    neighbors = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_a, descriptors_b, k=2)
    return [pair[0] for pair in neighbors
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


def match_images(image_a, image_b, geometry, settings):
    arrays = [np.asarray(image, dtype=np.uint8) for image in (image_a, image_b)]
    if any(array.ndim != 3 or array.shape[2] != 3 for array in arrays):
        raise ValueError("SIFT inputs must be RGB images")
    if geometry == "identity" and arrays[0].shape != arrays[1].shape:
        raise ValueError("Identity verification requires equal image dimensions")
    detector = cv2.SIFT_create(
        nfeatures=settings.nfeatures, contrastThreshold=settings.contrast_threshold,
        edgeThreshold=settings.edge_threshold, sigma=settings.sigma,
    )
    features = [detector.detectAndCompute(cv2.cvtColor(array, cv2.COLOR_RGB2GRAY), None)
                for array in arrays]
    (keypoints_a, descriptors_a), (keypoints_b, descriptors_b) = features
    forward = _ratio_matches(descriptors_a, descriptors_b, settings.ratio)
    matches = forward
    if settings.mutual:
        reverse = _ratio_matches(descriptors_b, descriptors_a, settings.ratio)
        reciprocal = {(match.trainIdx, match.queryIdx) for match in reverse}
        matches = [match for match in forward if (match.queryIdx, match.trainIdx) in reciprocal]
    matches = sorted(matches, key=lambda match: (match.queryIdx, match.trainIdx))
    a = np.array([keypoints_a[match.queryIdx].pt for match in matches], np.float32).reshape(-1, 2)
    b = np.array([keypoints_b[match.trainIdx].pt for match in matches], np.float32).reshape(-1, 2)
    mask, matrix, status = verify_geometry(a, b, geometry, settings)
    if not keypoints_a or not keypoints_b:
        status = "no_keypoints"
    count, verified = len(matches), int(mask.sum())
    metrics = {
        "keypoints_a": len(keypoints_a), "keypoints_b": len(keypoints_b),
        "ratio_matches_a_to_b": len(forward), "tentative_matches": count,
        "verified_matches": verified, "inlier_ratio": verified / count if count else 0.0,
        "geometry_status": status,
    }
    return MatchResult(metrics, a, b, np.array([m.distance for m in matches], np.float32), mask, matrix)


def match_panel(image_a, image_b, result, title, max_draw=0, display_side=520, seed=3407):
    """Only resize for display AFTER matching. Labels state any line-count cap."""
    images, scales = [], []
    for image in (image_a, image_b):
        size = (image.width, image.height)
        scale = min(1.0, display_side / max(size))
        resized = image.resize((max(1, round(size[0] * scale)), max(1, round(size[1] * scale))))
        images.append(resized)
        scales.append((resized.width / size[0], resized.height / size[1]))
    header, gap = 46, 10
    canvas = Image.new("RGB", (max(480, images[0].width + images[1].width + gap),
                                max(im.height for im in images) + header), "white")
    offsets = [(0, header), (images[0].width + gap, header)]
    for image, offset in zip(images, offsets):
        canvas.paste(image, offset)
    chosen = np.flatnonzero(result.inliers)
    if max_draw and len(chosen) > max_draw:
        chosen = np.sort(np.random.default_rng(seed).choice(chosen, max_draw, replace=False))
    draw = ImageDraw.Draw(canvas)
    colors = np.random.default_rng(seed).integers(40, 235, (len(chosen), 3))
    for index, color in zip(chosen, colors):
        ends = []
        for point, scale, offset in zip((result.points_a[index], result.points_b[index]), scales, offsets):
            ends.append((float(point[0] * scale[0] + offset[0]), float(point[1] * scale[1] + offset[1])))
        color = tuple(map(int, color))
        draw.line(ends, fill=color, width=1)
        for x, y in ends:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color)
    metrics = result.metrics
    draw.text((5, 4), title, fill="black")
    draw.text((5, 23), f"Verified: {metrics['verified_matches']} / {metrics['tentative_matches']}"
              f" | drawn: {len(chosen)} | {metrics['geometry_status']}", fill="black")
    return canvas


def stack_panels(panels):
    canvas = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    return canvas


def summarize(rows):
    """All pairs remain in count means; zero-match ratios are explicitly zero."""
    result = {}
    for condition in ("raw", "v16"):
        subset = [row for row in rows if row["condition"] == condition]
        tentative = sum(row["tentative_matches"] for row in subset)
        verified = sum(row["verified_matches"] for row in subset)
        result[condition] = {
            "pair_count": len(subset),
            "total_tentative_matches": tentative, "total_verified_matches": verified,
            "mean_tentative_matches": float(np.mean([r["tentative_matches"] for r in subset])),
            "mean_verified_matches": float(np.mean([r["verified_matches"] for r in subset])),
            "macro_inlier_ratio": float(np.mean([r["inlier_ratio"] for r in subset])),
            "micro_inlier_ratio": verified / tentative if tentative else 0.0,
            "zero_tentative_pairs": sum(r["tentative_matches"] == 0 for r in subset),
            "geometry_failure_pairs": sum(r["geometry_status"] != "ok" for r in subset),
        }
    indexed = {(r["pair_id"], r["condition"]): r for r in rows}
    deltas = [indexed[(r["pair_id"], "v16")]["verified_matches"] - r["verified_matches"]
              for r in rows if r["condition"] == "raw"]
    baseline = result["raw"]["total_verified_matches"]
    result["paired_change"] = {
        "mean_verified_match_delta": float(np.mean(deltas)),
        "increased_pairs": sum(d > 0 for d in deltas),
        "unchanged_pairs": sum(d == 0 for d in deltas),
        "decreased_pairs": sum(d < 0 for d in deltas),
        "total_verified_relative_gain": (sum(deltas) / baseline) if baseline else None,
    }
    return result


def protocol(settings, geometry):
    return {
        "version": "uie4-sift-1.0.0", "settings": asdict(settings), "geometry": geometry,
        "grayscale": "OpenCV RGB2GRAY on RGB uint8; no histogram equalization",
        "matcher": "BF L2 k=2; strict Lowe ratio; optional reciprocal ratio matches",
        "zero_matches": "Counts and macro/micro ratios are zero; pairs are never dropped",
        "interpretation": "Geometric consistency under the specified model, not match accuracy labels",
        "references": ["https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html",
                       "https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html"],
    }
