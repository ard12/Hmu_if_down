"""
2D Position Estimation via CSI Phase-Differential Triangulation.
Uses Weighted Least Squares (WLS) on Time Difference of Arrival (TDoA)
proxies derived from CSI phase slopes across >=3 receiver nodes.
"""

from collections import deque
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np


class PositionUncertainError(Exception):
    """Raised when position cannot be estimated due to insufficient or stale measurements."""
    pass


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm for point-in-polygon test (including edge boundary)."""
    n = len(poly)
    if n < 3:
        return False

    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        # Check if point is on segment
        if min(p1x, p2x) - 1e-7 <= x <= max(p1x, p2x) + 1e-7 and \
           min(p1y, p2y) - 1e-7 <= y <= max(p1y, p2y) + 1e-7:
            # Cross product to check collinearity on segment
            if abs((p2y - p1y) * (x - p1x) - (p2x - p1x) * (y - p1y)) < 1e-5:
                return True

        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def _dist_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _dist_to_polygon(px: float, py: float, poly: List[Tuple[float, float]]) -> float:
    """Minimum distance from point (px, py) to polygon boundary."""
    n = len(poly)
    min_d = float("inf")
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        d = _dist_to_segment(px, py, x1, y1, x2, y2)
        if d < min_d:
            min_d = d
    return min_d


class TriangulationEngine:
    """
    2D position estimation using CSI phase slope differentials across >=3 nodes.
    Algorithm: Weighted Least Squares (WLS).
    """

    MAX_MEASUREMENT_AGE_S = 2.0
    HISTORY_MAX_LEN = 60

    def __init__(self, node_positions: Dict[str, Tuple[float, float]]):
        """
        Args:
            node_positions: {node_id: (x_m, y_m)} floor plan coordinates in metres.
        """
        if len(node_positions) < 3:
            raise ValueError("Triangulation requires at least 3 node positions")

        self.node_positions = {str(k): (float(v[0]), float(v[1])) for k, v in node_positions.items()}
        self._measurements: Dict[str, dict] = {}
        self.floor_plan: Dict[str, List[Tuple[float, float]]] = {}
        self.history: deque = deque(maxlen=self.HISTORY_MAX_LEN)

    def set_floor_plan(self, rooms: Dict[str, List[Tuple[float, float]]]):
        """
        rooms: {room_id: [(x1, y1), (x2, y2), ...]}
        """
        self.floor_plan = {
            str(r_id): [(float(p[0]), float(p[1])) for p in poly]
            for r_id, poly in rooms.items()
        }

    def add_measurement(
        self,
        node_id: str,
        phase_slope: float,
        timestamp: Optional[float] = None,
        rssi: float = -50.0,
    ):
        """Ingest per-subcarrier CSI phase slope (proxy for distance)."""
        nid = str(node_id)
        ts = timestamp if timestamp is not None else time.time()
        self._measurements[nid] = {
            "phase_slope": float(phase_slope),
            "timestamp": float(ts),
            "rssi": float(rssi),
        }

    def _is_collinear(self, pts: List[Tuple[float, float]]) -> bool:
        """Check if active node positions are collinear (degenerate geometry)."""
        if len(pts) < 3:
            return True

        coords = np.array(pts)
        centered = coords - coords.mean(axis=0)
        _, s, _ = np.linalg.svd(centered)
        # If second singular value is near 0, points lie on a line
        return len(s) < 2 or s[1] < 1e-4

    def estimate_position(self, current_time: Optional[float] = None) -> dict:
        """
        Estimate 2D subject position via Weighted Least Squares (WLS).

        Returns:
          {
            "x": float,
            "y": float,
            "room_id": str,
            "confidence": float,
            "n_nodes_used": int
          }

        Raises:
          PositionUncertainError if < 3 fresh measurements.
        """
        now = current_time if current_time is not None else time.time()

        # Filter fresh measurements
        fresh_nodes = []
        for nid, (nx, ny) in self.node_positions.items():
            meas = self._measurements.get(nid)
            if meas is not None:
                age = now - meas["timestamp"]
                if age <= self.MAX_MEASUREMENT_AGE_S:
                    fresh_nodes.append((nid, nx, ny, meas["phase_slope"], meas["rssi"], age))

        if len(fresh_nodes) < 3:
            raise PositionUncertainError(
                f"Position uncertain: only {len(fresh_nodes)} fresh node measurements (need >= 3)"
            )

        pts = [(item[1], item[2]) for item in fresh_nodes]
        is_collinear = self._is_collinear(pts)

        # Build WLS system using node 0 as reference
        # Equation: 2(xi - x0)*x + 2(yi - y0)*y = (xi^2 + yi^2 - di^2) - (x0^2 + y0^2 - d0^2)
        ref_id, x0, y0, s0, rssi0, age0 = fresh_nodes[0]
        d0 = max(0.01, abs(s0))

        A_rows = []
        b_rows = []
        weights = []

        for nid, xi, yi, si, rssii, agei in fresh_nodes[1:]:
            di = max(0.01, abs(si))
            A_rows.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
            b_val = (xi * xi + yi * yi - di * di) - (x0 * x0 + y0 * y0 - d0 * d0)
            b_rows.append(b_val)
            # RSSI weight: 10^(rssi/20) with freshness penalty
            w = max(0.01, (10.0 ** (rssii / 20.0)) / (1.0 + agei))
            weights.append(w)

        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_rows, dtype=np.float64)
        W = np.diag(weights)

        if is_collinear:
            # Collinear degenerate geometry -> confidence 0
            x, y = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
            confidence = 0.0
        else:
            try:
                # Solve WLS: (A^T W A) p = A^T W b
                AtW = A.T @ W
                AtWA = AtW @ A
                AtWb = AtW @ b
                sol = np.linalg.solve(AtWA, AtWb)
                x = float(sol[0])
                y = float(sol[1])

                # Calculate confidence (0-1) based on residual error and measurement spread
                residuals = b - A @ sol
                mse = float(np.mean(residuals ** 2))
                confidence = max(0.05, min(1.0, 1.0 / (1.0 + 0.1 * mse)))
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to least squares
                sol, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
                x = float(sol[0])
                y = float(sol[1])
                confidence = 0.5

        # Determine room_id from floor plan
        room_id = "unknown"
        if self.floor_plan:
            # 1. Point in polygon test
            for r_id, poly in self.floor_plan.items():
                if _point_in_polygon(x, y, poly):
                    room_id = r_id
                    break

            # 2. If point is not inside any polygon (e.g. edge / hallway), find nearest room polygon
            if room_id == "unknown":
                best_dist = float("inf")
                best_room = "unknown"
                for r_id, poly in self.floor_plan.items():
                    d = _dist_to_polygon(x, y, poly)
                    if d < best_dist:
                        best_dist = d
                        best_room = r_id
                room_id = best_room

        result = {
            "x": round(x, 4),
            "y": round(y, 4),
            "room_id": room_id,
            "confidence": round(confidence, 4),
            "n_nodes_used": len(fresh_nodes),
        }

        self.history.append(result)
        return result

    def get_history(self) -> List[dict]:
        """Return history of position estimates (up to 60)."""
        return list(self.history)
