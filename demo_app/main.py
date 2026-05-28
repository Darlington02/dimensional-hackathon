from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import Any

import uvicorn

from demo_app.audio import AudioAlert
from demo_app.capture import CaptureBuffer
from demo_app.config import load_config
from demo_app.dashboard import create_app
from demo_app.detector import YoloToolDetector
from demo_app.patrol import PatrolController, nearest_waypoint
from demo_app.robot import Go2Runner
from demo_app.telegram_bot import TelegramBot
from demo_app.types import AlertEvent, Waypoint

logger = logging.getLogger(__name__)


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading config from config.yaml")
    cfg = load_config(Path("config.yaml"))
    logger.info("Config loaded")
    waypoints = [Waypoint(id=w.id, pos=tuple(w.pos), yaw_deg=w.yaw) for w in cfg.waypoints]

    logger.info("Creating dashboard state")
    web_state: dict[str, Any]
    command_handlers: dict[str, Any] = {}
    app, web_state = create_app(stream_fps=cfg.web.stream_fps, command_handlers=command_handlers)
    web_state["waypoints"] = [{"id": w.id, "pos": list(w.pos), "yaw": w.yaw_deg} for w in waypoints]

    runner = Go2Runner(
        robot_ip=cfg.robot.ip,
        obstacle_avoidance=cfg.robot.obstacle_avoidance,
        camera_resize=tuple(cfg.robot.camera_resize),
        connect_timeout_sec=cfg.robot.connect_timeout_sec,
    )
    logger.info("Starting Go2 runner")
    runner.start()
    logger.info("Go2 runner started")

    patrol = PatrolController(
        runner=runner,
        waypoints=waypoints,
        web_state=web_state,
        loop_forever=cfg.patrol.loop_forever,
        scan_turns=cfg.patrol.scan_turns,
        scan_pause_sec=cfg.patrol.scan_pause_sec,
    )

    capture = CaptureBuffer(
        buffer_seconds=cfg.alert.buffer_seconds,
        fps=cfg.alert.capture_fps,
        output_dir=Path("captures"),
    )
    audio = AudioAlert(audio_file=cfg.alert.audio_file, runner=runner)
    detector: YoloToolDetector | None = None

    async def on_patrol() -> None:
        await patrol.start()

    async def on_stop() -> None:
        await patrol.stop()

    async def on_status() -> str:
        return await patrol.status()

    async def command_forward() -> str:
        await patrol.stop()
        logger.info("Dashboard command: forward")
        ok = await patrol.manual_drive(forward_mps=0.25, duration_sec=1.0)
        return "ok" if ok else "failed"

    async def command_back() -> str:
        await patrol.stop()
        logger.info("Dashboard command: back")
        ok = await patrol.manual_drive(forward_mps=-0.20, duration_sec=1.0)
        return "ok" if ok else "failed"

    async def command_left() -> str:
        await patrol.stop()
        logger.info("Dashboard command: left")
        ok = await patrol.manual_drive(left_mps=0.15, duration_sec=1.0)
        return "ok" if ok else "failed"

    async def command_right() -> str:
        await patrol.stop()
        logger.info("Dashboard command: right")
        ok = await patrol.manual_drive(left_mps=-0.15, duration_sec=1.0)
        return "ok" if ok else "failed"

    async def command_turn_left() -> str:
        await patrol.stop()
        logger.info("Dashboard command: turn_left")
        ok = await patrol.manual_drive(yaw_radps=0.50, duration_sec=0.8)
        return "ok" if ok else "failed"

    async def command_turn_right() -> str:
        await patrol.stop()
        logger.info("Dashboard command: turn_right")
        ok = await patrol.manual_drive(yaw_radps=-0.50, duration_sec=0.8)
        return "ok" if ok else "failed"

    command_handlers.update(
        {
            "patrol": on_patrol,
            "stop": on_stop,
            "status": on_status,
            "forward": command_forward,
            "back": command_back,
            "left": command_left,
            "right": command_right,
            "turn_left": command_turn_left,
            "turn_right": command_turn_right,
        }
    )

    telegram = TelegramBot(
        token=cfg.telegram_bot_token,
        owner_chat_id=cfg.telegram_owner_chat_id,
        on_patrol_command=on_patrol,
        on_stop_command=on_stop,
        on_status_query=on_status,
    )

    def on_frame(frame: Any) -> None:
        ts = time.time()
        web_state["latest_frame"] = frame
        capture.push(frame, ts)

    video_sub = runner.video_stream().subscribe(on_next=on_frame)

    stop_event = asyncio.Event()
    last_pose_record = 0.0

    async def detection_loop() -> None:
        nonlocal last_pose_record
        nonlocal detector
        last_no_frame_log = 0.0
        last_no_detection_log = 0.0
        while not stop_event.is_set():
            await asyncio.sleep(cfg.detection.interval_sec)
            pose = runner.get_pose()
            web_state["robot_pose"] = {"x": pose.x, "y": pose.y, "yaw_deg": pose.yaw_deg}
            now = time.time()
            if now - last_pose_record >= 0.5:
                history = web_state.setdefault("pose_history", [])
                history.append([pose.x, pose.y])
                if len(history) > 400:
                    del history[: len(history) - 400]
                last_pose_record = now

            if not cfg.detection.enabled:
                continue
            if detector is None:
                logger.info("Initializing YOLO detector: %s", cfg.detection.model_name)
                detector = await asyncio.to_thread(
                    YoloToolDetector,
                    cfg.detection.detection_classes,
                    cfg.detection.conf_threshold,
                    cfg.detection.model_name,
                )
                logger.info(
                    "YOLO detector ready. Watching classes=%s conf_threshold=%.2f",
                    cfg.detection.detection_classes,
                    cfg.detection.conf_threshold,
                )

            frame = web_state.get("latest_frame")
            if frame is None:
                if now - last_no_frame_log >= 3.0:
                    logger.info("YOLO loop waiting for camera frames")
                    last_no_frame_log = now
                continue

            detections = await asyncio.to_thread(detector.detect, frame, time.time())
            if detections:
                logger.info(
                    "YOLO detected %d object(s): %s",
                    len(detections),
                    ", ".join(
                        f"{det.class_name}@{det.confidence:.2f}" for det in detections
                    ),
                )
            elif now - last_no_detection_log >= 3.0:
                logger.info("YOLO saw 0 watched objects in the current frame")
                last_no_detection_log = now
            web_state["latest_detections"] = [
                {
                    "bbox": list(det.bbox),
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "timestamp": det.timestamp,
                }
                for det in detections
            ]

            for det in detections:
                pose = runner.get_pose()
                waypoint_id = patrol.current_waypoint() or nearest_waypoint(pose, waypoints)
                if cfg.detection.cooldown_sec > 0:
                    logger.warning(
                        "Positive cooldown_sec is configured, but the current demo is set up "
                        "to capture every YOLO hit. Set cooldown in code if you want dedupe back."
                    )

                event = AlertEvent(
                    bbox=det.bbox,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    frame=det.frame,
                    robot_pose=(pose.x, pose.y),
                    nearest_waypoint=waypoint_id,
                    timestamp=det.timestamp,
                )
                web_state["anomalies"].append(
                    {
                        "class_name": event.class_name,
                        "robot_pose": list(event.robot_pose),
                        "nearest_waypoint": event.nearest_waypoint,
                        "timestamp": event.timestamp,
                    }
                )
                web_state["event_log"].append(
                    {"type": "alert", "payload": {"class_name": event.class_name}}
                )

                async def send_alert(ev: AlertEvent = event) -> None:
                    jpg_path = await capture.snapshot(ev)
                    web_state["event_log"].append(
                        {
                            "type": "capture_saved",
                            "payload": {
                                "class_name": ev.class_name,
                                "jpg": str(jpg_path),
                            },
                        }
                    )
                    logger.info("Saved detection JPG: %s (%s)", jpg_path, ev.class_name)
                    await telegram.send_photo_alert(ev, jpg_path)

                asyncio.create_task(send_alert())

    logger.info("Starting Telegram bot")
    await telegram.start()
    logger.info("Telegram bot started")
    web_state["event_log"].append(
        {
            "type": "telegram",
            "payload": {"message": f"owner_chat_id={cfg.telegram_owner_chat_id}"},
        }
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host=cfg.web.host, port=cfg.web.port, log_level="info", loop="asyncio")
    )
    logger.info("Starting web dashboard on http://%s:%s", cfg.web.host, cfg.web.port)
    server_task = asyncio.create_task(server.serve())
    detection_task = asyncio.create_task(detection_loop())

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: request_shutdown())

    await shutdown.wait()

    stop_event.set()
    detection_task.cancel()
    try:
        await detection_task
    except Exception:
        pass

    try:
        await patrol.stop()
    except Exception:
        pass

    try:
        video_sub.dispose()
    except Exception:
        pass

    try:
        await telegram.stop()
    except Exception:
        pass

    try:
        server.should_exit = True
        await server_task
    except Exception:
        pass

    runner.stop()


if __name__ == "__main__":
    asyncio.run(run())
