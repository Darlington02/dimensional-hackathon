from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class RobotConfig(BaseModel):
    ip: str = ""
    obstacle_avoidance: bool = True
    camera_resize: tuple[int, int] = (640, 480)
    connect_timeout_sec: float = 15.0


class WaypointConfig(BaseModel):
    id: str
    pos: tuple[float, float, float]
    yaw: float


class PatrolConfig(BaseModel):
    loop_forever: bool = True
    scan_turns: int = 4
    scan_pause_sec: float = 1.0


class DetectionConfig(BaseModel):
    enabled: bool = True
    model_name: str = "yoloe-26x-seg.pt"
    interval_sec: float = 0.5
    conf_threshold: float = 0.25
    cooldown_sec: int = 300
    detection_classes: list[str]


class AlertConfig(BaseModel):
    audio_file: str
    clip_duration_sec: int = 5
    buffer_seconds: int = 10
    capture_fps: int = 15


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    stream_fps: int = 10


class Settings(BaseModel):
    robot: RobotConfig
    waypoints: list[WaypointConfig]
    patrol: PatrolConfig
    detection: DetectionConfig
    alert: AlertConfig
    web: WebConfig
    telegram_bot_token: str
    telegram_owner_chat_id: int


def load_config(yaml_path: Path = Path("config.yaml")) -> Settings:
    load_dotenv()
    data = yaml.safe_load(yaml_path.read_text())
    data["telegram_bot_token"] = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    data["telegram_owner_chat_id"] = int(os.environ["TELEGRAM_OWNER_CHAT_ID"])

    robot_ip = os.environ.get("ROBOT_IP", "").strip()
    if robot_ip:
        data["robot"]["ip"] = robot_ip

    return Settings(**data)
