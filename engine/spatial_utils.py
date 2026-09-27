"""
Spatial utilities for SmartSpace AI:
- Oriented Bounding Box (OBB) geometry
- Separating Axis Theorem (SAT) 2D collision detection
- Boundary containment, clearance buffers, and ergonomic distance metrics
"""
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

def get_rotated_corners(cx: float, cy: float, w: float, d: float, angle_deg: float) -> np.ndarray:
    """
    Returns the 4 corner points of a rectangle centered at (cx, cy)
    with dimensions width (along local x) and depth (along local y),
    rotated by angle_deg (in degrees counter-clockwise).
    Corners returned in order: [Bottom-Left, Bottom-Right, Top-Right, Top-Left].
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    hw = w / 2.0
    hd = d / 2.0
    
    # Local unrotated offsets
    local_corners = np.array([
        [-hw, -hd],
        [ hw, -hd],
        [ hw,  hd],
        [-hw,  hd]
    ])
    
    # 2D Rotation matrix
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    rotated = local_corners @ R.T
    world_corners = rotated + np.array([cx, cy])
    return world_corners

def get_axes(corners: np.ndarray) -> List[np.ndarray]:
    """Returns normalized perpendicular axes for SAT from polygon corners."""
    axes = []
    num_pts = len(corners)
    for i in range(num_pts):
        p1 = corners[i]
        p2 = corners[(i + 1) % num_pts]
        edge = p2 - p1
        # Normal vector (-dy, dx)
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            axes.append(normal / norm)
    return axes

def project_onto_axis(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Projects corners onto a 2D axis and returns (min_val, max_val)."""
    dots = corners @ axis
    return float(np.min(dots)), float(np.max(dots))

def check_obb_collision(box_a_corners: np.ndarray, box_b_corners: np.ndarray, margin: float = 0.0) -> bool:
    """
    Checks if two oriented bounding boxes collide using Separating Axis Theorem (SAT).
    If margin > 0, tests whether they are closer than margin (expanded collision).
    """
    axes_a = get_axes(box_a_corners)
    axes_b = get_axes(box_b_corners)
    
    for axis in axes_a + axes_b:
        min_a, max_a = project_onto_axis(box_a_corners, axis)
        min_b, max_b = project_onto_axis(box_b_corners, axis)
        
        # If separated with margin, they do not collide
        if (max_a + margin < min_b) or (max_b + margin < min_a):
            return False
            
    return True

def is_within_room(cx: float, cy: float, w: float, d: float, angle_deg: float, 
                   room_w: float, room_l: float, wall_margin: float = 0.05) -> Tuple[bool, float]:
    """
    Checks if an oriented box is fully contained inside the room boundaries [0, room_w] x [0, room_l].
    Returns (is_inside, penalty_distance).
    """
    corners = get_rotated_corners(cx, cy, w, d, angle_deg)
    min_x, max_x = np.min(corners[:, 0]), np.max(corners[:, 0])
    min_y, max_y = np.min(corners[:, 1]), np.max(corners[:, 1])
    
    violation = 0.0
    if min_x < wall_margin:
        violation += (wall_margin - min_x)
    if max_x > room_w - wall_margin:
        violation += (max_x - (room_w - wall_margin))
    if min_y < wall_margin:
        violation += (wall_margin - min_y)
    if max_y > room_l - wall_margin:
        violation += (max_y - (room_l - wall_margin))
        
    return (violation <= 1e-4, violation)

def get_front_interaction_zone(cx: float, cy: float, w: float, d: float, angle_deg: float,
                               front_depth: float = 0.8) -> np.ndarray:
    """
    Returns the OBB corners for the ergonomic interaction/clearance zone
    in front of an item (e.g. bed edge, desk chair pullout, wardrobe doors opening).
    Local 'front' is +y direction before rotation.
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    # Distance from center to front edge + half front depth
    offset_dist = (d / 2.0) + (front_depth / 2.0)
    front_cx = cx - offset_dist * sin_a
    front_cy = cy + offset_dist * cos_a
    
    return get_rotated_corners(front_cx, front_cy, w, front_depth, angle_deg)

def calculate_wall_distance(cx: float, cy: float, w: float, d: float, angle_deg: float,
                            room_w: float, room_l: float) -> Tuple[float, str]:
    """
    Calculates the distance from the closest wall to the furniture back edge.
    Returns (min_distance, closest_wall_name: 'north'|'south'|'east'|'west').
    """
    corners = get_rotated_corners(cx, cy, w, d, angle_deg)
    min_x, max_x = np.min(corners[:, 0]), np.max(corners[:, 0])
    min_y, max_y = np.min(corners[:, 1]), np.max(corners[:, 1])
    
    dists = {
        "west": min_x,
        "east": room_w - max_x,
        "south": min_y,
        "north": room_l - max_y
    }
    
    closest_wall = min(dists, key=dists.get)
    return max(0.0, dists[closest_wall]), closest_wall

def check_door_clearance(furniture_corners: np.ndarray, door_info: Dict[str, Any],
                         room_w: float, room_l: float, swing_radius: float = 1.0) -> bool:
    """
    Checks if a furniture item overlaps the door's swing clearance arc/box.
    """
    wall = door_info.get("wall", "south")
    offset = door_info.get("offset", 0.5)
    dw = door_info.get("width", 0.9)
    
    # Calculate door clearance rectangle in world coordinates
    if wall == "south":
        door_box = np.array([
            [offset, 0.0],
            [offset + dw, 0.0],
            [offset + dw, swing_radius],
            [offset, swing_radius]
        ])
    elif wall == "north":
        door_box = np.array([
            [offset, room_l - swing_radius],
            [offset + dw, room_l - swing_radius],
            [offset + dw, room_l],
            [offset, room_l]
        ])
    elif wall == "west":
        door_box = np.array([
            [0.0, offset],
            [swing_radius, offset],
            [swing_radius, offset + dw],
            [0.0, offset + dw]
        ])
    else: # east
        door_box = np.array([
            [room_w - swing_radius, offset],
            [room_w, offset],
            [room_w, offset + dw],
            [room_w - swing_radius, offset + dw]
        ])
        
    return check_obb_collision(furniture_corners, door_box)

def check_window_overlap(furniture_corners: np.ndarray, windows: List[Dict[str, Any]],
                         room_w: float, room_l: float, clearance_depth: float = 0.5) -> bool:
    """
    Checks if furniture overlaps the immediate daylight zone in front of windows.
    """
    for win in windows:
        wall = win.get("wall", "north")
        offset = win.get("offset", 1.0)
        ww = win.get("width", 1.2)
        
        if wall == "north":
            win_box = np.array([
                [offset, room_l - clearance_depth],
                [offset + ww, room_l - clearance_depth],
                [offset + ww, room_l],
                [offset, room_l]
            ])
        elif wall == "south":
            win_box = np.array([
                [offset, 0.0],
                [offset + ww, 0.0],
                [offset + ww, clearance_depth],
                [offset, clearance_depth]
            ])
        elif wall == "west":
            win_box = np.array([
                [0.0, offset],
                [clearance_depth, offset],
                [clearance_depth, offset + ww],
                [0.0, offset + ww]
            ])
        else: # east
            win_box = np.array([
                [room_w - clearance_depth, offset],
                [room_w, offset],
                [room_w, offset + ww],
                [room_w - clearance_depth, offset + ww]
            ])
            
        if check_obb_collision(furniture_corners, win_box):
            return True
            
    return False
