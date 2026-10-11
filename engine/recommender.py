"""
SmartSpace AI — Dual-Engine Recommendation System
===================================================
Implements the budgeted furniture recommendation pipeline that powers the
``POST /api/recommendations`` endpoint.

Engine architecture
~~~~~~~~~~~~~~~~~~~
- **Primary engine (ML):** When a trained DQN checkpoint exists, its
  Q-network guides a short layout-refinement pass (up to 12 optimisation
  steps) on each candidate bundle.  The DQN is never labelled *trained*
  unless a real checkpoint file is present.
- **Secondary engine (rules):** Every candidate bundle — whether ML-refined
  or not — is validated through a strict filter chain:

  1. **SAT collision check** — No two furniture OBBs may overlap
     (calls ``engine.sat_collision.check_obb_collision``).
  2. **A* circulation** — The walking path from the entry door to every
     furniture interaction zone must be reachable at ≥ 75 % coverage.
  3. **Door swing / window daylight clearance** — No furniture may block
     the door arc or window light zone.
  4. **Room boundary** — All items must lie fully inside ``[0, W] × [0, L]``.
  5. **Budget cap** — Total furniture cost must stay within 90 % of the
     user's stated budget to leave room for decor.

Output
~~~~~~
Returns up to 6 style/value bundles, each carrying a fully laid-out room,
133-D MDP state vector, validation result, and preset provenance.

Connection to the rest of the project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Called by ``routes/recommendations.py``.  Delegates placement to
``engine.interior_env.InteriorEnv`` and collision math to
``engine.sat_collision``.  Reads the furniture catalog and preset layouts
from ``database.models``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from config import FURNITURE_SPECS
from engine.interior_env import InteriorEnv
from engine.spatial_utils import check_obb_collision, get_rotated_corners, is_within_room
from database.models import decode_json, list_furniture, list_presets


ROOM_ALIASES = {
    "living": "living_room", "living room": "living_room", "living_room": "living_room",
    "lounge": "living_room", "family room": "living_room",
    "bedroom": "bedroom", "regular bedroom": "bedroom", "guest bedroom": "bedroom",
    "master": "master_bedroom", "master bedroom": "master_bedroom", "master_bedroom": "master_bedroom",
    "primary bedroom": "master_bedroom",
    "office": "home_office", "home office": "home_office", "home_office": "home_office",
    "study": "home_office", "study room": "home_office", "work room": "home_office",
    "dining": "dining_room", "dining room": "dining_room", "dining_room": "dining_room",
    "kitchen": "kitchen", "bathroom": "bathroom", "bath": "bathroom", "washroom": "bathroom",
    "studio": "studio_apartment", "studio apartment": "studio_apartment", "studio_apartment": "studio_apartment",
    "studio flat": "studio_apartment",
}
ROOM_ITEMS = {
    "living_room": ["sofa", "tv", "table", "chair", "cabinet"],
    "bedroom": ["bed", "wardrobe", "desk", "chair", "cabinet"],
    "master_bedroom": ["bed", "wardrobe", "desk", "chair", "cabinet", "tv"],
    "home_office": ["desk", "chair", "cabinet", "cabinet", "chair"],
    "dining_room": ["table", "chair", "chair", "chair", "chair", "cabinet"],
    "kitchen": ["cabinet", "cabinet", "cabinet", "cabinet", "cabinet", "table"],
    "bathroom": ["cabinet", "cabinet"],
    "studio_apartment": ["bed", "desk", "chair", "wardrobe", "cabinet", "tv"],
}


def _canonical_room(value: str) -> str:
    clean = " ".join(str(value or "living_room").strip().lower().replace("_", " ").split())
    return ROOM_ALIASES.get(clean, clean.replace(" ", "_"))


def _style_key(style: str) -> str:
    clean = str(style or "modern").strip().lower()
    aliases = {"minimal": "minimalist", "scandi": "scandinavian"}
    return aliases.get(clean, clean)


def _choose_preset(room_type: str, width_m: float, length_m: float) -> Optional[Dict[str, Any]]:
    candidates = list_presets(room_type)
    if not candidates:
        return None
    requested_ratio = width_m / max(length_m, 0.001)

    def distance(preset: Dict[str, Any]) -> tuple[float, float]:
        preset_ratio = float(preset["room_width_m"]) / max(float(preset["room_length_m"]), 0.001)
        ratio_distance = math.sqrt((requested_ratio - preset_ratio) ** 2)
        return (ratio_distance, -float(preset["expert_ergonomic_score"]))

    return min(candidates, key=distance)


def _select_item(category: str, style: str, mode: str, remaining: float,
                 width_m: float, length_m: float,
                 role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    room_area = width_m * length_m
    products = list_furniture(category)
    if role:
        role_candidates = []
        if role == "ac":
            role_candidates = [p for p in products if "ac" in p["id"].lower() or "air conditioner" in p["name"].lower()]
        elif role == "bookshelf":
            role_candidates = [p for p in products if "book" in p["id"].lower() or "modular-storage" in p["id"].lower() or "tall-display" in p["id"].lower()]
        elif role == "sideboard":
            role_candidates = [p for p in products if "sideboard" in p["id"].lower() or "modular-storage" in p["id"].lower() or "tall-display" in p["id"].lower()]
        elif role == "kitchen_sink":
            role_candidates = [p for p in products if "kitchen-sink" in p["id"].lower() or ("sink" in p["id"].lower() and "vanity" not in p["id"].lower())]
        elif role == "kitchen_drawers":
            role_candidates = [p for p in products if "drawer" in p["id"].lower() or "spoon" in p["id"].lower() or "cutlery" in p["name"].lower()]
        elif role == "kitchen_stove":
            role_candidates = [p for p in products if "hob" in p["id"].lower() or "stove" in p["id"].lower() or "cooking" in p["name"].lower()]
        elif role == "kitchen_cupboard":
            role_candidates = [p for p in products if "cupboard" in p["id"].lower() or "overhead" in p["id"].lower() or "essential" in p["name"].lower()]
        elif role == "kitchen_counter":
            role_candidates = [p for p in products if "kitchen-counter" in p["id"].lower()]
        elif role == "kitchen_pantry":
            role_candidates = [p for p in products if "pantry" in p["id"].lower()]
        elif role == "kitchen_island":
            role_candidates = [p for p in products if "island" in p["id"].lower() or "prep" in p["name"].lower()]
        elif role == "bath_vanity":
            role_candidates = [p for p in products if "vanity" in p["id"].lower()]
        elif role == "bath_storage":
            role_candidates = [p for p in products if "wall-storage" in p["id"].lower()]
        elif role == "king_bed":
            role_candidates = [p for p in products if "king" in p["id"].lower()]
        elif role == "queen_bed":
            role_candidates = [p for p in products if "queen" in p["id"].lower()]
        elif role == "coffee_table":
            role_candidates = [p for p in products if "coffee" in p["id"].lower() or "central" in p["name"].lower()]
        elif role == "dining_table":
            role_candidates = [p for p in products if "dining" in p["id"].lower()]
        elif role == "lounge_chair":
            role_candidates = [p for p in products if "lounge" in p["id"].lower() or "armchair" in p["id"].lower() or "accent" in p["name"].lower()]
        elif role == "dining_chair":
            role_candidates = [p for p in products if "dining" in p["id"].lower()]
        elif role == "office_chair":
            role_candidates = [p for p in products if "desk" in p["id"].lower() or "ergonomic" in p["name"].lower() or "gaming" in p["id"].lower()]
        elif role == "visitor_chair":
            role_candidates = [p for p in products if "dining" in p["id"].lower() or "accent" in p["id"].lower()]
        elif role == "office_desk" or role == "study_desk":
            role_candidates = [p for p in products if "study" in p["id"].lower() or "writing" in p["id"].lower() or "executive" in p["id"].lower() or "standing" in p["id"].lower()]
        elif role == "wardrobe_large":
            role_candidates = [p for p in products if "3" in p["id"].lower() or "sliding" in p["id"].lower() or "tall" in p["id"].lower()]
        elif role == "wardrobe":
            role_candidates = [p for p in products if "sliding-2" in p["id"].lower() or "compact" in p["id"].lower() or "tall-wood" in p["id"].lower()]

        if role_candidates:
            budget_matches = [p for p in role_candidates if int(p["base_cost_inr"]) <= remaining]
            products = budget_matches if budget_matches else role_candidates

    products = [
        product for product in products
        if float(product["width"]) * float(product["depth"]) <= room_area * 0.35
        and (
            (float(product["width"]) <= width_m and float(product["depth"]) <= length_m)
            or (float(product["depth"]) <= width_m and float(product["width"]) <= length_m)
        )
        and int(product["base_cost_inr"]) <= remaining
    ]
    if not products:
        return None
    if mode == "value":
        products.sort(key=lambda item: (int(item["base_cost_inr"]), item["style_tag"] != style))
    else:
        products.sort(key=lambda item: (item["style_tag"] != style, int(item["base_cost_inr"])))
    return products[0]


def _default_position(category: str, index: int, width: float, length: float) -> tuple[float, float, int]:
    """Place common furniture types around a usable center route."""
    anchors = {
        "sofa": (width * .5, length * .78, 180),
        "tv": (width * .5, length * .12, 0),
        "bed": (width * .52, length * .70, 0),
        "wardrobe": (width * .12, length * .53, 90),
        "desk": (width * .77, length * .70, 0),
        "chair": (width * .72, length * .40, 0),
        "table": (width * .5, length * .47, 0),
        "cabinet": (width * .87, length * .5, 90),
    }
    x, y, rotation = anchors.get(category, (width * .5, length * .5, 0))
    if category == "chair" and index > 0:
        y = max(0.75, min(length - 0.75, y + (0.62 if index % 2 else -0.62)))
    if category == "cabinet" and index > 0:
        x = width * .14
    return x, y, rotation


def _separate_sat_overlaps(furniture: List[Dict[str, Any]], width: float,
                           length: float) -> None:
    """Resolve catalog-size changes while preserving each furniture SAT box."""
    priority = {"bed": 100, "sofa": 95, "tv": 90, "wardrobe": 80,
                "desk": 75, "table": 65, "cabinet": 60, "chair": 50}

    def corners(item: Dict[str, Any]):
        return get_rotated_corners(
            item["x"], item["y"], item["width"], item["depth"], item.get("rotation", 0)
        )

    def overlaps_others(index: int, candidate: Dict[str, Any]) -> bool:
        candidate_corners = corners(candidate)
        for other_index, other in enumerate(furniture):
            if other_index == index:
                continue
            if check_obb_collision(candidate_corners, corners(other), margin=0.05):
                return True
        return False

    def find_clear_position(index: int) -> Optional[tuple[float, float]]:
        item = furniture[index]
        step = 0.25
        xs = [round(i * step, 3) for i in range(1, int(width / step))]
        ys = [round(i * step, 3) for i in range(1, int(length / step))]
        options = sorted(
            ((x, y) for x in xs for y in ys),
            key=lambda point: ((point[0] - item["x"]) ** 2 + (point[1] - item["y"]) ** 2,
                               point[1], point[0]),
        )
        for x, y in options:
            candidate = {**item, "x": x, "y": y}
            inside, _ = is_within_room(
                x, y, item["width"], item["depth"], item.get("rotation", 0),
                width, length, wall_margin=0.04,
            )
            if inside and not overlaps_others(index, candidate):
                return x, y
        return None

    # Resolve one overlap at a time, moving the less essential piece first.
    for _ in range(len(furniture) * 2):
        pair = None
        for first in range(len(furniture)):
            for second in range(first + 1, len(furniture)):
                if check_obb_collision(corners(furniture[first]), corners(furniture[second]), margin=0.05):
                    pair = (first, second)
                    break
            if pair:
                break
        if not pair:
            return
        first, second = pair
        moving_order = sorted(
            (first, second),
            key=lambda index: (priority.get(furniture[index]["type"], 40), -index),
        )
        moved = False
        for index in moving_order:
            position = find_clear_position(index)
            if position:
                furniture[index]["x"], furniture[index]["y"] = position
                moved = True
                break
        if not moved:
            return


def _build_layout(items: List[Dict[str, Any]], preset: Dict[str, Any],
                  width: float, length: float, budget: float, style: str,
                  name: str, agent: Any = None) -> Dict[str, Any]:
    """Return a validated metric layout using an explicit, request-local agent."""
    packed = decode_json(preset.get("furniture_state_json"), {})
    source_layout = packed.get("layout", {})
    normalized = decode_json(preset.get("normalized_geometry_json"), [])
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for geometry in normalized:
        by_type.setdefault(geometry["type"], []).append(geometry)
    type_indexes: Dict[str, int] = {}
    furniture = []
    for index, product in enumerate(items):
        category = product["category"]
        type_index = type_indexes.get(category, 0)
        type_indexes[category] = type_index + 1
        geometry_list = by_type.get(category, [])
        if type_index < len(geometry_list):
            geometry = geometry_list[type_index]
            x = float(geometry["x"]) * width
            y = float(geometry["y"]) * length
            rotation = int(geometry.get("rotation", 0))
        else:
            x, y, rotation = _default_position(category, type_index, width, length)
            if preset["room_type"] == "dining_room" and category == "chair" and type_index >= 2:
                x, y, rotation = width * .5, length * .79, 180
        item_width = float(product["width"])
        item_depth = float(product["depth"])
        rotated = rotation % 180 != 0
        bound_w, bound_d = (item_depth, item_width) if rotated else (item_width, item_depth)
        x = min(max(x, bound_w / 2 + 0.04), width - bound_w / 2 - 0.04)
        y = min(max(y, bound_d / 2 + 0.04), length - bound_d / 2 - 0.04)
        furniture.append({
            "id": f"{category}_{index + 1}", "type": category, "label": product["name"],
            "x": round(x, 3), "y": round(y, 3), "width": item_width,
            "depth": item_depth, "height": float(product["height"]),
            "rotation": rotation, "cost": int(product["base_cost_inr"]),
            "preferred_wall": FURNITURE_SPECS[category].get("preferred_wall", False),
        })
    _separate_sat_overlaps(furniture, width, length)
    room = {
        "dimensions": {"width": width, "length": length},
        "budget": budget,
        "door": source_layout.get("door", {"wall": "south", "offset": 0.25, "width": 0.9}),
        "windows": source_layout.get("windows", []),
        "furniture": furniture,
    }
    env = InteriorEnv(room)
    model_refined = False
    if agent is not None and agent.is_trained:
        try:
            trajectory = agent.optimize_layout_trajectory(env, max_steps=12, time_budget_seconds=2.0)
            if trajectory:
                env.load_furniture(trajectory[-1]["layout"]["furniture"])
                model_refined = len(trajectory) > 1
        except Exception:
            # **Graceful degradation:** preserve the curated layout if the
            # optional learned refinement fails; SAT/A* still validate it below.
            import logging
            logging.getLogger(__name__).exception("DQN refinement failed; using curated layout")
            env = InteriorEnv(room)
    result = env.get_layout_dict()
    result.update({
        "name": name,
        "style": style,
        "room_type": preset["room_type"],
        "state_133d": env.get_state().tolist(),
    })
    metrics = result["metrics"]
    metrics.setdefault("total_cost", sum(float(item.get("cost", 0)) for item in furniture))
    metrics.setdefault("collision_count", 0)
    metrics.setdefault("door_interferences", 0)
    metrics.setdefault("window_blockages", 0)
    metrics.setdefault("boundary_violations", 0)
    metrics.setdefault("circulation_ratio", 0)
    metrics.setdefault("ergonomics_score", 100)
    tv_viewing_ok = _tv_sofa_ok(result["furniture"])
    valid = (
        metrics.get("collision_count", 0) == 0
        and metrics.get("boundary_violations", 0) == 0
        and metrics.get("door_interferences", 0) == 0
        and metrics.get("window_blockages", 0) == 0
        and metrics.get("circulation_ratio", 0) >= 0.75
        and metrics.get("total_cost", 0) <= budget * 0.90
        and tv_viewing_ok is not False
    )
    result["validation"] = {
        "valid": bool(valid),
        "engine": "trained_dqn" if model_refined else "curated_layout_rules",
        "model_refined": model_refined,
        "budget_cap_inr": round(budget * 0.90),
        "checks": {
            "sat_collisions": metrics.get("collision_count", 0),
            "door_swing_intersections": metrics.get("door_interferences", 0),
            "window_daylight_blockages": metrics.get("window_blockages", 0),
            "a_star_reachability": metrics.get("circulation_ratio", 0),
            "tv_sofa_viewing_range": tv_viewing_ok,
        },
    }
    return result


def _tv_sofa_ok(furniture: List[Dict[str, Any]]) -> Optional[bool]:
    sofas = [item for item in furniture if item["type"] == "sofa"]
    televisions = [item for item in furniture if item["type"] == "tv"]
    if not sofas or not televisions:
        return None
    return all(1.5 <= math.dist((sofa["x"], sofa["y"]), (tv["x"], tv["y"])) <= 3.8
               for sofa in sofas for tv in televisions)


def recommend_designs(payload: Dict[str, Any], agent: Any = None) -> Dict[str, Any]:
    """Return six style/value bundles with SAT/A*-checked layouts."""
    try:
        width_ft = float(payload.get("width_ft", payload.get("width", 14.0)))
        length_ft = float(payload.get("length_ft", payload.get("length", 16.0)))
        budget = float(payload.get("budget_inr", payload.get("budget", 85000)))
    except (TypeError, ValueError):
        raise ValueError("Width, length, and budget must be numbers.")
    if not (6 <= width_ft <= 60 and 6 <= length_ft <= 60):
        raise ValueError("Room width and length must be between 6 and 60 feet.")
    if not (5000 <= budget <= 100000000):
        raise ValueError("Budget must be between ₹5,000 and ₹10,00,00,000.")
    room_type = _canonical_room(payload.get("room_type", payload.get("roomType", "living_room")))
    if room_type not in ROOM_ITEMS:
        raise ValueError("Choose living room, bedroom, master bedroom, kitchen, bathroom, dining room, home office, or studio apartment.")
    style = _style_key(payload.get("style", payload.get("designStyle", "modern")))
    width_m = width_ft * 0.3048
    length_m = length_ft * 0.3048
    preset = _choose_preset(room_type, width_m, length_m)
    if not preset:
        raise ValueError("No starter layout is available for this room type yet.")
    budget_cap = budget * 0.90
    bundles = []
    variation_specs = [
        (style, "value", f"{style.title()} · Best value"),
        (style, "style", f"{style.title()} · Style match"),
    ]
    variation_specs.extend(
        (candidate_style, "style", f"{candidate_style.title()} edit")
        for candidate_style in ("modern", "minimalist", "scandinavian", "industrial", "contemporary")
        if candidate_style != style
    )

    for candidate_style, mode, label in variation_specs[:6]:
        spend = 0.0
        chosen = []
        skipped = []
        categories = ROOM_ITEMS[room_type]
        for idx, category in enumerate(categories):
            role = None
            if room_type == "master_bedroom":
                if category == "bed":
                    role = "king_bed"
                elif category == "wardrobe":
                    role = "wardrobe_large"
                elif category == "cabinet":
                    role = "ac"
                elif category == "chair":
                    role = "lounge_chair"
                elif category == "desk":
                    role = "study_desk"
                elif category == "tv":
                    role = "tv"
            elif room_type == "bedroom":
                if category == "bed":
                    role = "queen_bed"
                elif category == "wardrobe":
                    role = "wardrobe"
                elif category == "cabinet":
                    role = "ac"
                elif category == "desk":
                    role = "study_desk"
                elif category == "chair":
                    role = "desk_chair"
            elif room_type == "home_office":
                if category == "desk":
                    role = "office_desk"
                elif category == "chair" and idx == 1:
                    role = "office_chair"
                elif category == "chair":
                    role = "visitor_chair"
                elif category == "cabinet" and idx == 2:
                    role = "ac"
                elif category == "cabinet" and idx == 3:
                    role = "bookshelf"
            elif room_type == "dining_room":
                if category == "table":
                    role = "dining_table"
                elif category == "chair":
                    role = "dining_chair"
                elif category == "cabinet":
                    role = "sideboard"
            elif room_type == "kitchen":
                if category == "cabinet" and idx == 0:
                    role = "kitchen_sink"
                elif category == "cabinet" and idx == 1:
                    role = "kitchen_drawers"
                elif category == "cabinet" and idx == 2:
                    role = "kitchen_stove"
                elif category == "cabinet" and idx == 3:
                    role = "kitchen_cupboard"
                elif category == "cabinet" and idx == 4:
                    role = "kitchen_pantry"
                elif category == "cabinet":
                    role = "kitchen_counter"
                elif category == "table":
                    role = "kitchen_island"
            elif room_type == "bathroom":
                if category == "cabinet" and idx == 0:
                    role = "bath_vanity"
                elif category == "cabinet" and idx == 1:
                    role = "bath_storage"
            elif room_type == "studio_apartment":
                if category == "bed":
                    role = "queen_bed"
                elif category == "wardrobe":
                    role = "wardrobe"
                elif category == "desk":
                    role = "office_desk"
                elif category == "chair":
                    role = "office_chair"
                elif category == "cabinet":
                    role = "ac"
                elif category == "tv":
                    role = "tv"
            elif room_type == "living_room":
                if category == "table":
                    role = "coffee_table"
                elif category == "chair":
                    role = "lounge_chair"
                elif category == "cabinet":
                    role = "ac"
                elif category == "tv":
                    role = "tv"
            product = _select_item(category, candidate_style, mode, budget_cap - spend, width_m, length_m, role=role)
            if product is None:
                skipped.append(category)
                continue
            chosen.append(product)
            spend += float(product["base_cost_inr"])
        layout = _build_layout(chosen, preset, width_m, length_m, budget, candidate_style, label, agent=agent)
        metrics = layout["metrics"]
        is_complete = not skipped
        # A budget-limited partial bundle can still be a safe, usable layout;
        # the UI reports omitted categories so the user sees the trade-off.
        bundles.append({
            "id": f"{candidate_style}-{mode}", "name": label,
            "style": candidate_style, "items": [
                {"id": p["id"], "name": p["name"], "category": p["category"],
                 "width": p["width"], "depth": p["depth"],
                 "base_cost_inr": p["base_cost_inr"], "style_tag": p["style_tag"],
                 "thumbnail_url": p["thumbnail_url"]}
                for p in chosen
            ],
            "total_cost_inr": round(float(metrics.get("total_cost", 0))),
            "budget_cap_inr": round(budget_cap),
            "budget_used_percent": round(float(metrics.get("total_cost", 0)) / budget * 100, 1),
            "omitted_categories": sorted(set(skipped)),
            "complete": is_complete,
            "layout": layout,
            "metrics": metrics,
            "validation": layout["validation"],
            "preset_match": {
                "id": preset["id"], "name": preset["name"],
                "aspect_ratio_distance": round(abs(width_m / length_m - preset["room_width_m"] / preset["room_length_m"]), 4),
                "source": preset["source_benchmark"],
            },
        })
    result = {
        "success": True,
        "room_type": room_type,
        "dimensions": {"width_m": round(width_m, 2), "length_m": round(length_m, 2)},
        "budget_inr": round(budget), "bundle_budget_cap_inr": round(budget_cap),
        "requested_style": style,
        "primary_engine": {"name": "DQN layout policy", "available": bool(agent and agent.is_trained)},
        "active_generator": "trained_dqn" if any(bundle["validation"]["model_refined"] for bundle in bundles) else "curated_layout_rules",
        "secondary_engine": "SAT collision + A* circulation + clearance and budget validation",
        "variation_count": len(bundles),
        "bundles": bundles,
    }
    requested_variation = str(payload.get("variation_id") or "")
    if requested_variation:
        result["selected_bundle"] = next(
            (bundle for bundle in bundles if bundle["id"] == requested_variation), None
        )
    return result
