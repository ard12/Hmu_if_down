"""Radar point cloud 3D skeleton fitter using RANSAC and kinematic constraints."""

from typing import Any, Dict, List, Optional
import numpy as np


class SkeletonFitter:
    """
    Fits a simplified 5-segment skeleton (head, torso, l_arm, r_arm, legs)
    to a radar point cloud frame using RANSAC + kinematic constraints.

    Kinematic constraints:
      - Head centroid is 90–95% of max_z (standing height).
      - Torso segment: 50–90% of max_z.
      - Leg segment: 0–50% of max_z.
      - Arm segments: lateral extensions from torso centroid.
    """

    RANSAC_ITERATIONS: int = 50
    MIN_POINTS: int = 8  # minimum point cloud size for valid fit
    INLIER_DISTANCE_M: float = 0.35  # radius of torso/body cylinder

    def __init__(self, subject_height_m: float = 1.70):
        self.subject_height_m: float = float(subject_height_m)

    def fit(self, point_cloud: np.ndarray) -> Dict[str, Any]:
        """
        point_cloud: (N, 3) array of (x, y, z) in metres.
        Returns:
          {
            "head": [x, y, z],
            "torso_top": [x, y, z],
            "torso_bottom": [x, y, z],
            "left_wrist": [x, y, z] | None,
            "right_wrist": [x, y, z] | None,
            "fit_quality": float,    # 0–1 inlier ratio
            "valid": bool            # False if < MIN_POINTS or fit_quality < 0.4
          }
        """
        if not isinstance(point_cloud, np.ndarray):
            point_cloud = np.array(point_cloud, dtype=float)

        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3 or len(point_cloud) < self.MIN_POINTS:
            return {
                "head": [0.0, 0.0, 0.0],
                "torso_top": [0.0, 0.0, 0.0],
                "torso_bottom": [0.0, 0.0, 0.0],
                "left_wrist": None,
                "right_wrist": None,
                "fit_quality": 0.0,
                "valid": False,
            }

        n_pts = len(point_cloud)
        rng = np.random.RandomState(42)  # Deterministic seed for reproducible RANSAC

        best_inliers: List[int] = []
        best_line_pt: Optional[np.ndarray] = None
        best_line_dir: Optional[np.ndarray] = None

        # RANSAC fitting of the dominant body axis
        for _ in range(self.RANSAC_ITERATIONS):
            idx1, idx2 = rng.choice(n_pts, size=2, replace=False)
            p1, p2 = point_cloud[idx1], point_cloud[idx2]
            direction = p2 - p1
            dist = np.linalg.norm(direction)
            if dist < 0.05:
                continue
            u = direction / dist

            # Orthogonal distance from each point to the line (p1, u)
            diffs = point_cloud - p1
            projections = np.dot(diffs, u)[:, np.newaxis] * u
            ortho_dists = np.linalg.norm(diffs - projections, axis=1)

            inliers = np.where(ortho_dists <= self.INLIER_DISTANCE_M)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = list(inliers)
                best_line_pt = p1
                best_line_dir = u

        fit_quality = len(best_inliers) / float(n_pts) if n_pts > 0 else 0.0
        valid = (fit_quality >= 0.4 and len(best_inliers) >= self.MIN_POINTS)

        if not valid or len(best_inliers) == 0:
            return {
                "head": [0.0, 0.0, 0.0],
                "torso_top": [0.0, 0.0, 0.0],
                "torso_bottom": [0.0, 0.0, 0.0],
                "left_wrist": None,
                "right_wrist": None,
                "fit_quality": float(fit_quality),
                "valid": False,
            }

        inlier_pts = point_cloud[best_inliers]
        z_min = float(np.min(inlier_pts[:, 2]))
        z_max = float(np.max(inlier_pts[:, 2]))
        z_span = z_max - z_min

        x_min, x_max = float(np.min(inlier_pts[:, 0])), float(np.max(inlier_pts[:, 0]))
        x_span = x_max - x_min
        y_min, y_max = float(np.min(inlier_pts[:, 1])), float(np.max(inlier_pts[:, 1]))
        y_span = y_max - y_min

        # Check if subject is fallen (horizontal body orientation: low z span and low max z)
        is_horizontal = (z_max <= 0.55 and z_span <= 0.45) or (z_span < 0.5 and max(x_span, y_span) >= 0.7)

        if is_horizontal:
            # Person is lying down / fallen horizontally
            # Major horizontal axis determines head vs feet
            if x_span >= y_span:
                # Main axis along X
                y_c = float(np.mean(inlier_pts[:, 1]))
                z_c = float(np.mean(inlier_pts[:, 2]))
                scale = self.subject_height_m / 1.70

                # Head at the higher or lower X end, torso in middle
                head_x = x_max
                head = [round(head_x, 3), round(y_c, 3), round(min(z_max, 0.35), 3)]
                torso_top = [round(head_x - 0.25 * scale, 3), round(y_c, 3), round(z_c, 3)]
                torso_bottom = [round(head_x - 0.70 * scale, 3), round(y_c, 3), round(z_c, 3)]
            else:
                x_c = float(np.mean(inlier_pts[:, 0]))
                z_c = float(np.mean(inlier_pts[:, 2]))
                scale = self.subject_height_m / 1.70

                head_y = y_max
                head = [round(x_c, 3), round(head_y, 3), round(min(z_max, 0.35), 3)]
                torso_top = [round(x_c, 3), round(head_y - 0.25 * scale, 3), round(z_c, 3)]
                torso_bottom = [round(x_c, 3), round(head_y - 0.70 * scale, 3), round(z_c, 3)]

            left_wrist = None
            right_wrist = None
        else:
            # Person is standing or sitting upright
            x_c = float(np.mean(inlier_pts[:, 0]))
            y_c = float(np.mean(inlier_pts[:, 1]))

            # Scale segments proportionally with subject_height_m
            h_effective = max(z_max, 0.1)
            # Scaling factor based on ratio to baseline 1.70m
            scale = self.subject_height_m / 1.70

            # Head centroid: 90-95% of standing height (or max_z)
            head_z = round(0.925 * h_effective * (self.subject_height_m / 1.70 if abs(h_effective - 1.70) < 0.2 else 1.0), 3)
            # If input cloud was scaled by subject_height, directly use proportional heights
            head_z = round(0.925 * z_max, 3)
            torso_top_z = round(0.80 * z_max, 3)
            torso_bottom_z = round(0.50 * z_max, 3)

            head = [round(x_c, 3), round(y_c, 3), head_z]
            torso_top = [round(x_c, 3), round(y_c, 3), torso_top_z]
            torso_bottom = [round(x_c, 3), round(y_c, 3), torso_bottom_z]

            # Detect arm extensions (points with lateral deviation from torso centroid)
            left_pts = inlier_pts[(inlier_pts[:, 0] < x_c - 0.15) & (inlier_pts[:, 2] >= 0.4 * z_max)]
            right_pts = inlier_pts[(inlier_pts[:, 0] > x_c + 0.15) & (inlier_pts[:, 2] >= 0.4 * z_max)]

            left_wrist = [round(float(v), 3) for v in np.mean(left_pts, axis=0)] if len(left_pts) >= 2 else None
            right_wrist = [round(float(v), 3) for v in np.mean(right_pts, axis=0)] if len(right_pts) >= 2 else None

        return {
            "head": head,
            "torso_top": torso_top,
            "torso_bottom": torso_bottom,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "fit_quality": round(float(fit_quality), 3),
            "valid": bool(valid),
        }

    def track(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Fit skeleton to a sequence of point cloud frames."""
        return [self.fit(frame) for frame in frames]
