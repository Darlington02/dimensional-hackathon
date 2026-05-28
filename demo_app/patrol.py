from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from demo_app.types import Pose, Waypoint

logger = logging.getLogger(__name__)


class PatrolController:
    SPEED_MPS = 0.5

    def __init__(
        self,
        runner,
        waypoints: list[Waypoint],
        web_state: dict[str, Any],
        loop_forever: bool = True,
        scan_turns: int = 4,
        scan_pause_sec: float = 1.0,
    ) -> None:
        self._runner = runner
        self._waypoints = waypoints
        self._web_state = web_state
        self._loop_forever = loop_forever
        self._scan_turns = scan_turns
        self._scan_pause_sec = scan_pause_sec
        self._task: asyncio.Task | None = None
        self._stop_requested = False
        self._current_wp: str | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_requested = False
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_requested = True
        task = self._task
        if task is not None:
            await task

    async def move_relative(self, forward: float, left: float, degrees: float) -> bool:
        pose = self._runner.get_pose()
        yaw_rad = math.radians(pose.yaw_deg)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        target_x = pose.x + cos_y * forward - sin_y * left
        target_y = pose.y + sin_y * forward + cos_y * left
        target_yaw = pose.yaw_deg + degrees
        dist = math.hypot(target_x - pose.x, target_y - pose.y)
        duration = max(0.75, dist / self.SPEED_MPS) if dist > 0.05 else max(0.75, abs(degrees) / 90.0)
        self._web_state["planned_path"] = [[pose.x, pose.y], [target_x, target_y]]
        self._push_event(
            "move",
            {"forward": forward, "left": left, "degrees": degrees},
        )
        return await asyncio.to_thread(
            self._runner.move_to,
            target_x,
            target_y,
            target_yaw,
            duration,
        )

    async def manual_drive(
        self,
        *,
        forward_mps: float = 0.0,
        left_mps: float = 0.0,
        yaw_radps: float = 0.0,
        duration_sec: float = 0.75,
    ) -> bool:
        self._push_event(
            "move",
            {
                "forward_mps": forward_mps,
                "left_mps": left_mps,
                "yaw_radps": yaw_radps,
                "duration_sec": duration_sec,
            },
        )
        return await asyncio.to_thread(
            self._runner.drive_for_duration,
            forward_mps,
            left_mps,
            yaw_radps,
            duration_sec,
        )

    async def status(self) -> str:
        mode = self._web_state.get("mode", "IDLE")
        visited = len(self._web_state.get("visited", []))
        current = self._current_wp or "-"
        return f"mode={mode} current={current} visited={visited}"

    def current_waypoint(self) -> str:
        return self._current_wp or ""

    async def _run(self) -> None:
        self._set_mode("PATROLLING")
        self._web_state["visited"] = []
        self._push_event("state", {"mode": "PATROLLING"})

        try:
            while not self._stop_requested:
                for waypoint in self._waypoints:
                    if self._stop_requested:
                        break
                    self._current_wp = waypoint.id
                    await self._move_to_waypoint(waypoint)
                    self._web_state["visited"].append(waypoint.id)
                    await self._scan_in_place()

                if not self._loop_forever:
                    break

            self._set_mode("RETURNING")
            self._push_event("state", {"mode": "RETURNING"})
            await self._return_home()
        finally:
            self._current_wp = None
            self._web_state["planned_path"] = []
            self._set_mode("IDLE")
            self._push_event("state", {"mode": "IDLE"})

    async def _move_to_waypoint(self, waypoint: Waypoint) -> None:
        pose = self._runner.get_pose()
        self._web_state["planned_path"] = [[pose.x, pose.y], [waypoint.pos[0], waypoint.pos[1]]]
        dist = math.hypot(waypoint.pos[0] - pose.x, waypoint.pos[1] - pose.y)
        duration = max(1.0, dist / self.SPEED_MPS)
        logger.info("Moving to %s", waypoint.id)
        await asyncio.to_thread(
            self._runner.move_to,
            waypoint.pos[0],
            waypoint.pos[1],
            waypoint.yaw_deg,
            duration,
        )

    async def _scan_in_place(self) -> None:
        for _ in range(self._scan_turns):
            if self._stop_requested:
                return
            pose = self._runner.get_pose()
            await asyncio.to_thread(self._runner.move_to, pose.x, pose.y, pose.yaw_deg + 90.0, 1.0)
            await asyncio.sleep(self._scan_pause_sec)

    async def _return_home(self) -> None:
        pose = self._runner.get_pose()
        self._web_state["planned_path"] = [[pose.x, pose.y], [0.0, 0.0]]
        await asyncio.to_thread(self._runner.move_to, 0.0, 0.0, 0.0, 5.0)

    def _set_mode(self, mode: str) -> None:
        self._web_state["mode"] = mode

    def _push_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._web_state.setdefault("event_log", []).append({"type": event_type, "payload": payload})


def nearest_waypoint(pose: Pose, waypoints: list[Waypoint]) -> str:
    best_id = ""
    best_dist = float("inf")
    for waypoint in waypoints:
        dist = (pose.x - waypoint.pos[0]) ** 2 + (pose.y - waypoint.pos[1]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = waypoint.id
    return best_id
