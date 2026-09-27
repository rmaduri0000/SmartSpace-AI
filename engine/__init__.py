"""SmartSpace AI Optimization Engine Package"""
from engine.spatial_utils import (
    get_rotated_corners, check_obb_collision, is_within_room,
    check_door_clearance, check_window_overlap
)
from engine.astar_planner import AStarPlanner
from engine.interior_env import InteriorEnv
from engine.dqn_agent import DQNAgent

__all__ = [
    "get_rotated_corners",
    "check_obb_collision",
    "is_within_room",
    "check_door_clearance",
    "check_window_overlap",
    "AStarPlanner",
    "InteriorEnv",
    "DQNAgent"
]
