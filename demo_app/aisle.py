from __future__ import annotations

import numpy as np

from demo_app.types import AisleObservation


class AisleCorridorDetector:
    def __init__(
        self,
        x_min_m: float,
        x_max_m: float,
        half_width_m: float,
        min_points_in_zone: int,
    ) -> None:
        self._x_min = x_min_m
        self._x_max = x_max_m
        self._half_width = half_width_m
        self._min_points = min_points_in_zone

    def analyze(self, points: np.ndarray) -> AisleObservation:
        if points.size == 0:
            return AisleObservation(
                corridor_clear=True,
                obstruction_distance_m=None,
                obstruction_direction="center",
                obstruction_point_count=0,
                obstruction_center_xy=None,
            )

        pts = points.astype(np.float32, copy=False)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.size == 0:
            return AisleObservation(
                corridor_clear=True,
                obstruction_distance_m=None,
                obstruction_direction="center",
                obstruction_point_count=0,
                obstruction_center_xy=None,
            )

        corridor_mask = (
            (pts[:, 0] >= self._x_min)
            & (pts[:, 0] <= self._x_max)
            & (np.abs(pts[:, 1]) <= self._half_width)
        )
        zone = pts[corridor_mask]
        point_count = int(zone.shape[0])
        if point_count < self._min_points:
            return AisleObservation(
                corridor_clear=True,
                obstruction_distance_m=None,
                obstruction_direction="center",
                obstruction_point_count=point_count,
                obstruction_center_xy=None,
            )

        nearest_idx = int(np.argmin(zone[:, 0]))
        nearest_distance = float(zone[nearest_idx, 0])
        center_y = float(np.mean(zone[:, 1]))
        if center_y > 0.12:
            direction = "left"
        elif center_y < -0.12:
            direction = "right"
        else:
            direction = "center"

        center_x = float(np.mean(zone[:, 0]))
        return AisleObservation(
            corridor_clear=False,
            obstruction_distance_m=nearest_distance,
            obstruction_direction=direction,
            obstruction_point_count=point_count,
            obstruction_center_xy=(center_x, center_y),
        )
