"""
SmartSpace AI - Global Configuration and Domain Constants
Defines YOLO 10-class taxonomy, furniture physical dimensions, ergonomic rules,
budget approximations, and room templates.
"""
from typing import Dict, List, Any

# 10 Target Classes for YOLO Interior Detection (as specified in project requirements)
TARGET_CLASSES = [
    "bed",       # 0
    "sofa",      # 1
    "chair",     # 2
    "table",     # 3
    "wardrobe",  # 4
    "desk",      # 5
    "tv",        # 6
    "cabinet",   # 7
    "door",      # 8
    "window"     # 9
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(TARGET_CLASSES)}

# Palette for bounding boxes and 2D/3D visualization
CLASS_COLORS = {
    "bed": "#3b82f6",       # Blue
    "sofa": "#8b5cf6",      # Purple
    "chair": "#06b6d4",     # Cyan
    "table": "#10b981",     # Emerald
    "wardrobe": "#f59e0b",  # Amber
    "desk": "#6366f1",      # Indigo
    "tv": "#ec4899",        # Pink
    "cabinet": "#f97316",   # Orange
    "door": "#ef4444",      # Red
    "window": "#14b8a6"     # Teal
}

# Standard real-world physical furniture specifications (Width x Depth x Height in meters)
# and estimated retail cost (in Indian rupees)
FURNITURE_SPECS: Dict[str, Dict[str, Any]] = {
    "bed": {
        "width": 1.6, "depth": 2.0, "height": 0.8,
        "base_cost": 12000, "clearance_front": 0.8, "clearance_sides": 0.6,
        "category": "furniture", "preferred_wall": True, "label": "Queen Bed"
    },
    "sofa": {
        "width": 2.1, "depth": 0.9, "height": 0.85,
        "base_cost": 14000, "clearance_front": 0.9, "clearance_sides": 0.3,
        "category": "furniture", "preferred_wall": False, "label": "3-Seater Sofa"
    },
    "chair": {
        "width": 0.6, "depth": 0.6, "height": 0.85,
        "base_cost": 1500, "clearance_front": 0.5, "clearance_sides": 0.2,
        "category": "furniture", "preferred_wall": False, "label": "Accent / Dining Chair"
    },
    "table": {
        "width": 1.4, "depth": 0.8, "height": 0.75,
        "base_cost": 3500, "clearance_front": 0.7, "clearance_sides": 0.7,
        "category": "furniture", "preferred_wall": False, "label": "Dining / Coffee Table"
    },
    "wardrobe": {
        "width": 1.5, "depth": 0.6, "height": 2.1,
        "base_cost": 8500, "clearance_front": 0.9, "clearance_sides": 0.2,
        "category": "furniture", "preferred_wall": True, "label": "2-Door Wardrobe"
    },
    "desk": {
        "width": 1.3, "depth": 0.65, "height": 0.75,
        "base_cost": 4200, "clearance_front": 0.8, "clearance_sides": 0.3,
        "category": "furniture", "preferred_wall": True, "label": "Study / Work Desk"
    },
    "tv": {
        "width": 1.2, "depth": 0.15, "height": 0.7,
        "base_cost": 8000, "clearance_front": 1.8, "clearance_sides": 0.2,
        "category": "electronics", "preferred_wall": True, "label": "55\" Wall / Console TV"
    },
    "cabinet": {
        "width": 1.0, "depth": 0.45, "height": 0.9,
        "base_cost": 3500, "clearance_front": 0.7, "clearance_sides": 0.2,
        "category": "furniture", "preferred_wall": True, "label": "Side Storage Cabinet"
    },
    "door": {
        "width": 0.9, "depth": 0.15, "height": 2.1,
        "base_cost": 0, "clearance_front": 1.0, "clearance_sides": 0.2,
        "category": "architectural", "preferred_wall": True, "label": "Entry Door"
    },
    "window": {
        "width": 1.4, "depth": 0.15, "height": 1.2,
        "base_cost": 0, "clearance_front": 0.6, "clearance_sides": 0.2,
        "category": "architectural", "preferred_wall": True, "label": "Window (Natural Light)"
    }
}

# Ergonomic & Architectural Layout Rules (referenced from Interior Design literature & paper)
ERGONOMIC_RULES = {
    "min_walkway_width": 0.75,         # Minimum clearance aisle between any objects (meters)
    "optimal_walkway_width": 0.90,     # Desired main circulation path width
    "door_swing_clearance": 0.95,      # Radius in front of door that must remain 100% unobstructed
    "window_light_clearance": 0.60,    # Clearance depth in front of windows to prevent daylight blockage
    "tv_min_viewing_distance": 1.50,   # Minimum distance between sofa/bed and TV
    "tv_max_viewing_distance": 3.80,   # Maximum comfortable viewing distance
    "desk_window_light_bonus": 1.5,    # Ergonomic bonus for placing desk with lateral or front natural daylight
    "bed_wall_adherence_bonus": 2.0,   # Bonus for headboard firmly placed along a wall
    "wardrobe_wall_adherence_bonus": 2.0, # Wardrobes must back against a wall
    "collision_penalty_scale": -15.0,  # Heavy negative reward for physical intersections
    "boundary_penalty_scale": -20.0,   # Heavy penalty for placing items outside room perimeter
    "connectivity_reward_scale": 10.0  # Reward when all furniture interaction zones are reachable via A*
}

# Pre-configured sample room profiles for rapid demonstration & verification
SAMPLE_ROOMS = {
    "living_room": {
        "id": "living_room",
        "name": "Modern Living Room",
        "room_type": "living_room",
        "dimensions": {"width": 5.4, "length": 4.8, "height": 2.8},
        "budget": 85000,
        "door": {"wall": "south", "offset": 0.3, "width": 0.9},
        "windows": [{"wall": "north", "offset": 1.7, "width": 1.5}],
        "initial_furniture": [
            {"id": "sofa_1", "type": "sofa", "x": 3.35, "y": 3.88, "rotation": 0, "label": "3-Seater Sofa"},
            {"id": "table_1", "type": "table", "x": 3.35, "y": 2.55, "rotation": 0, "label": "Coffee Table"},
            {"id": "tv_1", "type": "tv", "x": 3.35, "y": 0.45, "rotation": 0, "label": "55\" TV Console"},
            {"id": "chair_1", "type": "chair", "x": 1.35, "y": 2.65, "rotation": 0, "label": "Accent Lounge Chair"},
            {"id": "cabinet_1", "type": "cabinet", "x": 0.48, "y": 3.88, "rotation": 90, "label": "Split Air Conditioner (AC)"}
        ]
    },
    "bedroom": {
        "id": "bedroom",
        "name": "Cozy Bedroom",
        "room_type": "bedroom",
        "dimensions": {"width": 4.8, "length": 4.4, "height": 2.8},
        "budget": 65000,
        "door": {"wall": "south", "offset": 0.25, "width": 0.9},
        "windows": [{"wall": "north", "offset": 1.65, "width": 1.5}],
        "initial_furniture": [
            {"id": "bed_1", "type": "bed", "x": 2.7, "y": 3.1, "rotation": 0, "label": "Queen Bed"},
            {"id": "wardrobe_1", "type": "wardrobe", "x": 0.55, "y": 1.6, "rotation": 90, "label": "2-Door Wardrobe"},
            {"id": "desk_1", "type": "desk", "x": 4.0, "y": 1.6, "rotation": 270, "label": "Study Table"},
            {"id": "chair_1", "type": "chair", "x": 3.25, "y": 1.6, "rotation": 90, "label": "Desk Chair"},
            {"id": "cabinet_1", "type": "cabinet", "x": 0.45, "y": 3.7, "rotation": 90, "label": "Split Air Conditioner (AC)"}
        ]
    },
    "master_bedroom": {
        "id": "master_bedroom",
        "name": "Contemporary Master Bedroom",
        "room_type": "master_bedroom",
        "dimensions": {"width": 5.4, "length": 4.8, "height": 2.8},
        "budget": 95000,
        "door": {"wall": "south", "offset": 0.8, "width": 0.9},
        "windows": [{"wall": "north", "offset": 1.7, "width": 1.8}],
        "initial_furniture": [
            {"id": "bed_1", "type": "bed", "x": 3.1, "y": 3.6, "rotation": 0, "label": "King Size Bed"},
            {"id": "wardrobe_1", "type": "wardrobe", "x": 0.45, "y": 2.2, "rotation": 90, "label": "3-Door Wardrobe"},
            {"id": "cabinet_1", "type": "cabinet", "x": 0.4, "y": 4.1, "rotation": 90, "label": "Split Air Conditioner (AC)"},
            {"id": "desk_1", "type": "desk", "x": 4.8, "y": 1.8, "rotation": 270, "label": "Study / Dressing Table"},
            {"id": "chair_1", "type": "chair", "x": 4.0, "y": 1.8, "rotation": 90, "label": "Accent Chair"},
            {"id": "tv_1", "type": "tv", "x": 3.1, "y": 0.35, "rotation": 0, "label": "55\" Wall TV"}
        ]
    },
    "home_office": {
        "id": "home_office",
        "name": "Ergonomic Home Office",
        "room_type": "home_office",
        "dimensions": {"width": 4.2, "length": 3.8, "height": 2.7},
        "budget": 52000,
        "door": {"wall": "south", "offset": 0.25, "width": 0.9},
        "windows": [{"wall": "north", "offset": 1.1, "width": 1.4}],
        "initial_furniture": [
            {"id": "desk_1", "type": "desk", "x": 2.7, "y": 2.8, "rotation": 0, "label": "Study Table / Workstation"},
            {"id": "chair_1", "type": "chair", "x": 2.7, "y": 1.95, "rotation": 180, "label": "Ergonomic Office Chair"},
            {"id": "cabinet_1", "type": "cabinet", "x": 0.48, "y": 3.1, "rotation": 90, "label": "Split Air Conditioner (AC)"},
            {"id": "cabinet_2", "type": "cabinet", "x": 0.48, "y": 1.5, "rotation": 90, "label": "Bookshelf Storage"},
            {"id": "chair_2", "type": "chair", "x": 3.6, "y": 1.6, "rotation": 270, "label": "Visitor Chair"}
        ]
    },
    "dining_room": {
        "id": "dining_room",
        "name": "Contemporary Dining Room",
        "room_type": "dining_room",
        "dimensions": {"width": 4.8, "length": 4.3, "height": 2.8},
        "budget": 55000,
        "door": {"wall": "south", "offset": 0.25, "width": 0.9},
        "windows": [{"wall": "north", "offset": 1.4, "width": 1.5}],
        "initial_furniture": [
            {"id": "table_1", "type": "table", "x": 2.3, "y": 2.3, "rotation": 0, "label": "Dining Table"},
            {"id": "chair_1", "type": "chair", "x": 1.25, "y": 2.3, "rotation": 90, "label": "Dining Chair"},
            {"id": "chair_2", "type": "chair", "x": 3.35, "y": 2.3, "rotation": 270, "label": "Dining Chair"},
            {"id": "chair_3", "type": "chair", "x": 2.3, "y": 1.35, "rotation": 180, "label": "Dining Chair"},
            {"id": "chair_4", "type": "chair", "x": 2.3, "y": 3.25, "rotation": 0, "label": "Dining Chair"},
            {"id": "cabinet_1", "type": "cabinet", "x": 4.0, "y": 3.85, "rotation": 0, "label": "Sideboard Cabinet"}
        ]
    },
    "kitchen": {
        "id": "kitchen",
        "name": "Modern Kitchen",
        "room_type": "kitchen",
        "dimensions": {"width": 4.2, "length": 3.8, "height": 2.8},
        "budget": 60000,
        "door": {"wall": "south", "offset": 0.25, "width": 0.9},
        "windows": [{"wall": "north", "offset": 1.3, "width": 1.4}],
        "initial_furniture": [
            {"id": "cabinet_1", "type": "cabinet", "x": 0.45, "y": 2.0, "rotation": 90, "label": "Modular Kitchen Counter"},
            {"id": "cabinet_2", "type": "cabinet", "x": 3.75, "y": 2.0, "rotation": 270, "label": "Pantry Storage Cabinet"},
            {"id": "table_1", "type": "table", "x": 2.1, "y": 3.2, "rotation": 0, "label": "Kitchen Prep Island"}
        ]
    },
    "bathroom": {
        "id": "bathroom",
        "name": "Minimal Bathroom",
        "room_type": "bathroom",
        "dimensions": {"width": 3.2, "length": 2.8, "height": 2.5},
        "budget": 28000,
        "door": {"wall": "south", "offset": 0.25, "width": 0.8},
        "windows": [{"wall": "north", "offset": 1.0, "width": 1.0}],
        "initial_furniture": [
            {"id": "cabinet_1", "type": "cabinet", "x": 0.45, "y": 1.95, "rotation": 90, "label": "Vanity Counter & Basin"},
            {"id": "cabinet_2", "type": "cabinet", "x": 2.75, "y": 1.95, "rotation": 270, "label": "Wall Storage Unit"}
        ]
    },
    "studio_apartment": {
        "id": "studio_apartment",
        "name": "Urban Studio Apartment",
        "room_type": "studio_apartment",
        "dimensions": {"width": 5.4, "length": 4.8, "height": 2.8},
        "budget": 85000,
        "door": {"wall": "south", "offset": 0.8, "width": 0.9},
        "windows": [{"wall": "north", "offset": 1.7, "width": 1.8}],
        "initial_furniture": [
            {"id": "bed_1", "type": "bed", "x": 1.5, "y": 3.6, "rotation": 0, "label": "Queen Studio Bed"},
            {"id": "wardrobe_1", "type": "wardrobe", "x": 0.45, "y": 1.8, "rotation": 90, "label": "Compact Wardrobe"},
            {"id": "desk_1", "type": "desk", "x": 4.5, "y": 3.4, "rotation": 0, "label": "Study Table & Workstation"},
            {"id": "chair_1", "type": "chair", "x": 4.5, "y": 2.5, "rotation": 180, "label": "Ergonomic Chair"},
            {"id": "cabinet_1", "type": "cabinet", "x": 5.0, "y": 4.4, "rotation": 270, "label": "Split Air Conditioner (AC)"},
            {"id": "tv_1", "type": "tv", "x": 3.2, "y": 0.35, "rotation": 0, "label": "55\" Wall TV"}
        ]
    }
}
