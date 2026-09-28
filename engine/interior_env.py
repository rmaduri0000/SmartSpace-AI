"""
Gymnasium-compatible Interior Layout Environment for SmartSpace AI
Formulates furniture layout optimization as a Markov Decision Process (MDP)
directly referencing research paper specifications with multi-objective rewards:
- Collision avoidance (SAT)
- A* circulation connectivity
- Ergonomic alignments (bed to wall, sofa-to-TV viewing distance, desk natural light)
- Door and window daylight clearance
- Budget adherence
"""
import copy
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from config import TARGET_CLASSES, CLASS_TO_IDX, FURNITURE_SPECS, ERGONOMIC_RULES
from engine.spatial_utils import (
    get_rotated_corners,
    check_obb_collision,
    is_within_room,
    get_front_interaction_zone,
    calculate_wall_distance,
    check_door_clearance,
    check_window_overlap
)
from engine.astar_planner import AStarPlanner

class InteriorEnv:
    """
    Interior Design Reinforcement Learning Environment.
    """
    def __init__(self, room_config: Optional[Dict[str, Any]] = None):
        self.room_config = room_config or {}
        dims = self.room_config.get("dimensions", {})
        self.room_w = float(dims.get("width", self.room_config.get("room_width", 4.8)))
        self.room_l = float(dims.get("length", self.room_config.get("room_length", 4.0)))
        self.budget = float(self.room_config.get("budget", 40000.0))
        self.door = self.room_config.get("door", {"wall": "south", "offset": 0.8, "width": 0.9})
        self.windows = self.room_config.get("windows", [{"wall": "north", "offset": 1.5, "width": 1.5}])
        
        # Door world-space entry coordinates
        self.door_pos = self._compute_door_entry_coords()
        
        # Max number of furniture items handled in state vector
        self.max_items = 8
        self.furniture: List[Dict[str, Any]] = []
        self.planner = AStarPlanner(self.room_w, self.room_l, grid_res=0.15)
        
        # Action space definition:
        # For each furniture item:
        # 0: Shift +X (+0.25m)
        # 1: Shift -X (-0.25m)
        # 2: Shift +Y (+0.25m)
        # 3: Shift -Y (-0.25m)
        # 4: Rotate +90 deg
        # 5: Snap to closest wall
        self.actions_per_item = 6
        self.action_dim = self.max_items * self.actions_per_item
        
        # State vector: [room_w, room_l, budget_ratio, door_x, door_y,
        #                for each item: [norm_x, norm_y, norm_w, norm_d, norm_rot, class_onehot(10), placed]]
        # Dimension = 5 + max_items * (5 + 10 + 1) = 5 + 8 * 16 = 133
        self.item_feature_dim = 16
        self.state_dim = 5 + self.max_items * self.item_feature_dim
        
        self.step_count = 0
        self.max_steps = 50
        self.current_reward = 0.0
        
        items = self.room_config.get("furniture") or self.room_config.get("initial_furniture")
        if items:
            self.load_furniture(items)
            
    def _compute_door_entry_coords(self) -> Tuple[float, float]:
        wall = self.door.get("wall", "south")
        offset = self.door.get("offset", 0.8)
        dw = self.door.get("width", 0.9)
        cx = offset + dw / 2.0
        
        if wall == "south":
            return (cx, 0.4)
        elif wall == "north":
            return (cx, self.room_l - 0.4)
        elif wall == "west":
            return (0.4, offset + dw / 2.0)
        else: # east
            return (self.room_w - 0.4, offset + dw / 2.0)

    def load_furniture(self, furniture_list: List[Dict[str, Any]]):
        """Loads and normalizes furniture specifications."""
        self.furniture = []
        for i, item in enumerate(furniture_list[:self.max_items]):
            f_type = item.get("type", "chair").lower()
            specs = FURNITURE_SPECS.get(f_type, FURNITURE_SPECS["chair"])
            
            f_item = {
                "id": item.get("id", f"{f_type}_{i+1}"),
                "type": f_type,
                "x": float(item.get("x", self.room_w / 2.0)),
                "y": float(item.get("y", self.room_l / 2.0)),
                "width": float(item.get("width", specs["width"])),
                "depth": float(item.get("depth", specs["depth"])),
                "height": float(item.get("height", specs["height"])),
                "rotation": int(item.get("rotation", 0)) % 360,
                "cost": float(item.get("cost", specs["base_cost"])),
                "preferred_wall": specs.get("preferred_wall", False)
            }
            self.furniture.append(f_item)

    def reset(self) -> np.ndarray:
        """Resets the environment and returns the state representation."""
        self.step_count = 0
        items = self.room_config.get("furniture") or self.room_config.get("initial_furniture")
        if items:
            self.load_furniture(items)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Encodes the current room, door, and furniture configurations
        into a normalized numerical tensor for DQN input.
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Room context (normalized)
        state[0] = self.room_w / 10.0
        state[1] = self.room_l / 10.0
        state[2] = self.budget / 10000.0
        state[3] = self.door_pos[0] / max(1.0, self.room_w)
        state[4] = self.door_pos[1] / max(1.0, self.room_l)
        
        offset = 5
        for i in range(self.max_items):
            if i < len(self.furniture):
                item = self.furniture[i]
                c_idx = CLASS_TO_IDX.get(item["type"], 2)
                
                # Position & dimensions normalized
                state[offset + 0] = item["x"] / max(1.0, self.room_w)
                state[offset + 1] = item["y"] / max(1.0, self.room_l)
                state[offset + 2] = item["width"] / 3.0
                state[offset + 3] = item["depth"] / 3.0
                state[offset + 4] = (item["rotation"] % 360) / 360.0
                
                # One-hot class encoding (10 classes)
                state[offset + 5 + c_idx] = 1.0
                # Placed flag
                state[offset + 15] = 1.0
            offset += self.item_feature_dim
            
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one discrete action chosen by the DQN agent.
        Returns: (next_state, reward, done, info)
        """
        self.step_count += 1
        num_items = len(self.furniture)
        
        if num_items == 0:
            return self.get_state(), 0.0, True, {"msg": "No furniture to place"}
            
        item_idx = (action // self.actions_per_item) % num_items
        act_type = action % self.actions_per_item
        
        target = self.furniture[item_idx]
        delta_dist = 0.25 # 25cm movements
        
        # Apply action
        if act_type == 0:   # +X
            target["x"] = min(self.room_w - target["width"]/2, target["x"] + delta_dist)
        elif act_type == 1: # -X
            target["x"] = max(target["width"]/2, target["x"] - delta_dist)
        elif act_type == 2: # +Y
            target["y"] = min(self.room_l - target["depth"]/2, target["y"] + delta_dist)
        elif act_type == 3: # -Y
            target["y"] = max(target["depth"]/2, target["y"] - delta_dist)
        elif act_type == 4: # Rotate +90 deg
            target["rotation"] = (target["rotation"] + 90) % 360
        elif act_type == 5: # Snap to closest wall
            dist, wall = calculate_wall_distance(
                target["x"], target["y"], target["width"], target["depth"],
                target["rotation"], self.room_w, self.room_l
            )
            hw = target["width"] / 2.0
            hd = target["depth"] / 2.0
            if wall == "south":
                target["y"] = hd + 0.05
            elif wall == "north":
                target["y"] = self.room_l - hd - 0.05
            elif wall == "west":
                target["x"] = hw + 0.05
            elif wall == "east":
                target["x"] = self.room_w - hw - 0.05
                
        # Evaluate layout reward
        eval_metrics = self.evaluate_layout()
        reward = eval_metrics["total_reward"]
        
        done = self.step_count >= self.max_steps
        info = {
            "step": self.step_count,
            "metrics": eval_metrics,
            "modified_item": target["id"],
            "action_type": act_type
        }
        
        return self.get_state(), reward, done, info

    def evaluate_layout(self) -> Dict[str, Any]:
        """
        Computes the research paper's multi-objective reward formulation:
        - Collision Penalty (Inter-furniture SAT collisions)
        - Boundary Containment Penalty
        - Door Swing Interference Penalty
        - Window Daylight Clearance Penalty
        - Wall Alignment Reward (Bed, Wardrobe, Desk against walls)
        - Ergonomic TV-to-Sofa / Bed Viewing Distance Reward
        - A* Circulation Path Reachability Reward
        - Budget Adherence
        """
        num_items = len(self.furniture)
        if num_items == 0:
            return {"total_reward": 0.0, "ergonomics_score": 100}
            
        collision_count = 0
        boundary_violations = 0.0
        door_interferences = 0
        window_blockages = 0
        wall_alignment_score = 0.0
        ergonomic_bonuses = 0.0
        
        corners_list = []
        for item in self.furniture:
            corners = get_rotated_corners(
                item["x"], item["y"], item["width"], item["depth"], item["rotation"]
            )
            corners_list.append(corners)
            
            # 1. Boundary check
            inside, dist_out = is_within_room(
                item["x"], item["y"], item["width"], item["depth"], item["rotation"],
                self.room_w, self.room_l
            )
            if not inside:
                boundary_violations += dist_out
                
            # 2. Door swing interference
            if check_door_clearance(corners, self.door, self.room_w, self.room_l):
                door_interferences += 1
                
            # 3. Window daylight blockage (for tall items: wardrobe, high cabinet)
            if item["type"] in ["wardrobe", "cabinet"] and check_window_overlap(corners, self.windows, self.room_w, self.room_l):
                window_blockages += 1
                
            # 4. Wall alignment for items that prefer backing a wall
            if item.get("preferred_wall", False):
                min_dist, _ = calculate_wall_distance(
                    item["x"], item["y"], item["width"], item["depth"],
                    item["rotation"], self.room_w, self.room_l
                )
                if min_dist < 0.20:
                    wall_alignment_score += 1.0
                else:
                    wall_alignment_score -= (min_dist * 0.5)

        # 5. Pairwise inter-furniture collision check via SAT
        for i in range(num_items):
            for j in range(i + 1, num_items):
                if check_obb_collision(corners_list[i], corners_list[j], margin=0.05):
                    collision_count += 1

        # 6. Ergonomic rules (e.g. Sofa facing TV)
        sofa_items = [it for it in self.furniture if it["type"] == "sofa"]
        tv_items = [it for it in self.furniture if it["type"] == "tv"]
        for s in sofa_items:
            for t in tv_items:
                dist = math.hypot(s["x"] - t["x"], s["y"] - t["y"])
                if ERGONOMIC_RULES["tv_min_viewing_distance"] <= dist <= ERGONOMIC_RULES["tv_max_viewing_distance"]:
                    ergonomic_bonuses += 2.0
                else:
                    ergonomic_bonuses -= 1.0

        # 7. A* Circulation evaluation
        self.planner.build_occupancy_grid(self.furniture, inflation_margin=0.08)
        circ_eval = self.planner.evaluate_circulation(self.door_pos, self.furniture)
        circulation_ratio = circ_eval["circulation_score"]
        
        # 8. Budget evaluation
        total_cost = sum(item.get("cost", 0.0) for item in self.furniture)
        budget_penalty = max(0.0, (total_cost - self.budget) / 1000.0)
        
        # Reward composition weights
        r_collision = -15.0 * collision_count
        r_boundary = -20.0 * boundary_violations
        r_door = -25.0 * door_interferences
        r_window = -10.0 * window_blockages
        r_wall = 2.0 * wall_alignment_score
        r_ergo = 1.5 * ergonomic_bonuses
        r_circ = 12.0 * circulation_ratio
        r_budget = -5.0 * budget_penalty
        
        total_reward = (
            r_collision + r_boundary + r_door + r_window +
            r_wall + r_ergo + r_circ + r_budget
        )
        
        # Overall Ergonomic & Quality Score (0 to 100%)
        # Deduct for collisions, door blocks, and unreachable items
        score = 100.0
        score -= min(50.0, collision_count * 25.0)
        score -= min(30.0, door_interferences * 30.0)
        score -= min(20.0, boundary_violations * 20.0)
        score -= (1.0 - circulation_ratio) * 20.0
        score = max(0.0, min(100.0, score + (wall_alignment_score * 3.0)))
        
        return {
            "total_reward": round(float(total_reward), 2),
            "ergonomics_score": round(float(score), 1),
            "collision_count": collision_count,
            "door_interferences": door_interferences,
            "boundary_violations": round(float(boundary_violations), 2),
            "circulation_ratio": round(float(circulation_ratio), 2),
            "unreachable_count": len(circ_eval["unreachable_items"]),
            "paths": circ_eval["paths"],
            "total_cost": total_cost,
            "budget": self.budget
        }

    def get_layout_dict(self) -> Dict[str, Any]:
        """Returns clean serializable layout for frontend rendering."""
        eval_metrics = self.evaluate_layout()
        return {
            "room_width": self.room_w,
            "room_length": self.room_l,
            "door": self.door,
            "door_pos": self.door_pos,
            "windows": self.windows,
            "budget": self.budget,
            "furniture": copy.deepcopy(self.furniture),
            "metrics": eval_metrics
        }
