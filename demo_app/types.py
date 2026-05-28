from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw_deg: float


@dataclass(frozen=True)
class Waypoint:
    id: str
    pos: tuple[float, float, float]
    yaw_deg: float


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    frame: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class AlertEvent:
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    frame: np.ndarray
    robot_pose: tuple[float, float]
    nearest_waypoint: str
    timestamp: float
