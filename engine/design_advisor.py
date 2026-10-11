"""
SmartSpace AI — Rule-Based Interior Design Advisor
====================================================
Generates comprehensive, professional interior design recommendations covering:
1. Spatial feasibility and circulation clearances (A*, SAT collision, door/window zones)
2. Harmonious color palette direction (60-30-10 rule, wood species, metal accents)
3. Three-tier architectural lighting schemes (ambient, task, accent)
4. Room-specific designer pro tips and ergonomic standards
5. Finishing layers, acoustic softening, and budget allocation
"""
from typing import Any, Dict, List


STYLE_ACCENTS = {
    "modern": "matte black hardware and brushed brass accents",
    "minimalist": "subtle tone-on-tone neutrals with brushed stainless steel",
    "scandinavian": "pale natural timber, matte white, and soft sage green accents",
    "industrial": "matte gunmetal, burnished cognac leather, and blackened steel",
    "contemporary": "warm champagne brass, brushed graphite, and deep forest green accents",
}

STYLE_WOODS = {
    "modern": "natural white oak or clean walnut",
    "minimalist": "bleached ash or light birch",
    "scandinavian": "pale beech, blonde pine, or light oak",
    "industrial": "reclaimed rustic timber, dark mango, or distressed oak",
    "contemporary": "rich American walnut or wire-brushed fumed oak",
}

ROOM_LIGHTING = {
    "living_room": "Layer dimmable 2700K ambient ceiling downlights with a sculptural floor lamp by seating. Install warm indirect bias lighting behind the TV console to prevent eye strain during evening movie viewing.",
    "bedroom": "Prioritize warm 2700K ambient illumination with independent bedside reading sconces mounted at 105 cm from the floor. Keep ceiling lights on dimmers and avoid glaring fixtures directly above the mattress pillows.",
    "master_bedroom": "Design a 3-zone luxury master lighting scheme: soft dimmable cove ambient lighting, dedicated directional task lighting inside the wardrobe/dressing corridor, and hotel-style bedside pendant lights paired with warm reading spots.",
    "home_office": "Combine high-CRI (90+) 4000K daylight task illumination (500+ lux on work surface) with an adjustable arm lamp. Position desk perpendicular to the window to avoid monitor glare and illuminate video calls evenly.",
    "dining_room": "Center a warm pendant or chandelier 75–85 cm (30–34\") above the dining table top. Complement with soft perimeter wall sconces (2700K) on a separate dimmer to create an intimate ambiance for evening gatherings.",
    "kitchen": "Install high-output 3500K–4000K LED task strip lighting directly underneath upper wall cabinets to illuminate prep countertops shadow-free. Pair with glare-free ceiling downlights and pendant lights over the prep island.",
    "bathroom": "Flank the vanity mirror with vertical wall sconces on both sides at eye level (~165 cm) to eliminate unflattering face shadows. Use IP44+ moisture-rated fixtures and soft indirect cove lighting for nighttime visits.",
    "studio_apartment": "Implement flexible zoned switching so the sleeping corner can be dimmed separately from the workstation and living nook. Utilize directional floor lamps and under-shelf LEDs to define micro-spaces without rigid walls.",
    "default": "Layer general ambient ceiling illumination (2700K–3000K) with focused task lighting and warm accent lamps to give the room dimension and flexibility.",
}

ROOM_FINISHES = {
    "living_room": "Anchor seating with an area rug large enough that front legs of the sofa and accent chairs rest firmly on it. Balance smooth leather or wood with tactile bouclé, woven linen cushions, and double-pleat linen curtains.",
    "bedroom": "Place a plush wool rug extending at least 65 cm around the sides and foot of the bed. Install dual curtain tracks (sheers for filtered daylight privacy and blackout drapery for restorative sleep).",
    "master_bedroom": "Introduce a statement upholstered headboard, an 8x10 ft textured wool rug under the King bed, velvet accent cushions, and custom wardrobe hardware in brushed brass or graphite.",
    "home_office": "Install a low-pile acoustic rug to dampen sound and keyboard reverberation while allowing chair casters to roll smoothly. Incorporate concealed cable conduit channels along the desk base for a clean, distraction-free environment.",
    "dining_room": "Choose a low-pile or flatweave rug that extends at least 65 cm beyond table edges so pulled-back chairs remain fully on the rug. Repeat one timber finish between table and sideboard.",
    "kitchen": "Select stain-resistant porcelain or quartz surfaces with minimal grout lines. Organize spoons, cutlery, and cooking tools in dedicated partitioned drawers, keep the wash basin counter sealed, and mount cupboards overhead for quick access to daily kitchen essentials.",
    "bathroom": "Repeat a single metal finish (matte black, brushed brass, or chrome) across faucets, towel hooks, and cabinet hardware. Use moisture-resistant marine ply or slatted teak for warm natural accents.",
    "studio_apartment": "Maintain consistent flooring throughout to visually enlarge the apartment. Use translucent linen curtains as light dividers and choose nesting tables or lift-top storage furniture.",
    "default": "Repeat one accent color in two small details, such as cushions and artwork, to make the room feel cohesive and intentional.",
}

ROOM_DESIGNER_TIPS = {
    "living_room": "Maintain 45 cm (18\") clearance between the coffee table and sofa edge for comfortable legroom and easy reach. Center the TV console at seated eye level (~105 cm) and maintain 1.8–2.4 m viewing distance.",
    "bedroom": "Position the bed against a solid wall in the 'command position' facing the door without being in direct line of drafts. Ensure at least 60–75 cm (24–30\") walkway clearance on both sides of the mattress.",
    "master_bedroom": "Allocate 90 cm (36\") clearance in front of wardrobes for full-swing doors and dressing comfort. Center the King bed on the main architectural wall with symmetric 50–60 cm bedside approach corridors.",
    "home_office": "Follow the 90-90-90 ergonomic standard (elbows, hips, and knees at 90 degrees with feet flat). Ensure 85–100 cm behind the desk for chair rollout and clear access to storage units.",
    "dining_room": "Maintain minimum 90 cm (36\") clearance between table edges and walls or sideboards so guests can pull chairs out and pass behind comfortably without turning sideways.",
    "kitchen": "Honor the kitchen work triangle connecting the wash basin sink, cooking stove hob, and pantry storage. Position cutlery and spoon drawers directly beside the prep station, mount daily essentials cupboards overhead within arm's reach, and maintain 105–120 cm walkway clearance around the central island.",
    "bathroom": "Standard vanity height is 85–90 cm. Allow at least 75 cm clearance in front of the vanity basin and position the mirror 15–20 cm above the washbasin faucet.",
    "studio_apartment": "Zone functional territories without physical partitions: use an area rug to anchor living, a directional desk lamp to define work, and an open bookshelf room divider for sleeping privacy while preserving sightlines.",
    "default": "Preserve 75 cm minimum walkways between major furniture groupings to maintain uninterrupted circulation throughout the room.",
}


def _room_key(room_type: str) -> str:
    value = (room_type or "").strip().lower()
    if "master" in value:
        return "master_bedroom"
    if "studio" in value:
        return "studio_apartment"
    if "bed" in value:
        return "bedroom"
    if "office" in value or "work" in value or "study" in value:
        return "home_office"
    if "dining" in value:
        return "dining_room"
    if "kitchen" in value:
        return "kitchen"
    if "bath" in value or "wash" in value:
        return "bathroom"
    return "living_room"


def build_interior_recommendations(
    room: Dict[str, Any], metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Create comprehensive layout, palette, lighting, architectural, and finishing suggestions."""
    room_type = str(room.get("room_type") or room.get("roomType") or "Living Room")
    room_key = _room_key(room_type)
    room_display_name = room_type.replace("_", " ").title()
    style = str(room.get("style") or room.get("designStyle") or "Modern").strip()
    if not style:
        style = "Modern"
    style_key = style.lower()
    accent = STYLE_ACCENTS.get(style_key, STYLE_ACCENTS["modern"])
    wood = STYLE_WOODS.get(style_key, STYLE_WOODS["modern"])
    primary = str(room.get("primaryColor") or "warm white").strip()
    secondary = str(room.get("secondaryColor") or "soft neutral").strip()
    material = str(room.get("material") or room.get("preferredMaterial") or "natural wood").strip()

    issues: List[str] = []
    collisions = int(metrics.get("collision_count", 0) or 0)
    door_blocks = int(metrics.get("door_interferences", 0) or 0)
    window_blocks = int(metrics.get("window_blockages", 0) or 0)
    boundary_violations = float(metrics.get("boundary_violations", 0) or 0)
    circulation = float(metrics.get("circulation_ratio", 1.0) or 0)
    unreachable = int(metrics.get("unreachable_count", 0) or 0)
    if collisions:
        issues.append(f"separate the {collisions} overlapping furniture pair(s)")
    if door_blocks:
        issues.append("clear the door opening and swing clearance area")
    if window_blocks:
        issues.append(f"move tall units away from {window_blocks} blocked window daylight cone(s)")
    if boundary_violations > 0:
        issues.append("shift furniture fully inside room walls")
    if circulation < 0.999:
        count_text = f" ({unreachable} item(s) currently unreachable from the entry)" if unreachable else ""
        issues.append(f"open a 75 cm walking route from the entrance{count_text}")

    if issues:
        if circulation < 0.999:
            layout_title = "Clear the main circulation route"
        elif collisions:
            layout_title = "Resolve overlapping furniture collisions"
        elif door_blocks:
            layout_title = "Protect doorway swing clearance"
        elif window_blocks:
            layout_title = "Preserve natural window daylight"
        else:
            layout_title = "Keep furniture inside room boundaries"
        layout_detail = "Spatial priority: " + "; ".join(issues) + ". Recheck the green A* walking paths after adjusting pieces."
    else:
        layout_title = "Preserve ergonomic walkways"
        layout_detail = f"The current {room_display_name} passes SAT collision, room-boundary, doorway swing, and A* circulation checks. Preserve the entry path as you add decor."

    budget = float(room.get("budget", metrics.get("budget", 0)) or 0)
    total_cost = float(metrics.get("total_cost", 0) or 0)
    remaining = budget - total_cost
    if budget and remaining < 0:
        budget_detail = f"Current furniture estimate is INR {total_cost:,.0f}, which is INR {abs(remaining):,.0f} over budget. Consider smaller furniture sizes before adding lighting and decor."
    elif budget:
        budget_detail = f"About INR {remaining:,.0f} remains from your INR {budget:,.0f} budget. Allocate ~60% of this for architectural lighting and ~40% for soft furnishings and rugs."
    else:
        budget_detail = f"Current furniture is estimated at INR {total_cost:,.0f}. Keep a 15–20% separate allowance for architectural lighting, curtains, and styling accessories."

    recommendations = [
        {
            "category": "Layout & Circulation",
            "title": layout_title,
            "detail": layout_detail,
        },
        {
            "category": "Color & Materials",
            "title": f"Harmonized {style} Palette (60-30-10)",
            "detail": f"Follow the 60-30-10 interior rule: 60% dominant {primary} across walls and ceilings, 30% secondary {secondary} in primary upholstery and cabinetry, and 10% {accent} in metal finishes and styling decor. Anchor with {wood} and tactile {material} surfaces.",
        },
        {
            "category": "Architectural Lighting",
            "title": f"3-Tier Lighting Scheme for {room_display_name}",
            "detail": ROOM_LIGHTING.get(room_key, ROOM_LIGHTING["default"]),
        },
        {
            "category": "Designer Pro Tips",
            "title": f"Architectural Guidelines for {room_display_name}",
            "detail": ROOM_DESIGNER_TIPS.get(room_key, ROOM_DESIGNER_TIPS["default"]),
        },
        {
            "category": "Finishing & Budget",
            "title": "Textiles, Rugs & Finishing Touches",
            "detail": f"{ROOM_FINISHES.get(room_key, ROOM_FINISHES['default'])} {budget_detail}",
        },
    ]
    return {
        "style": style,
        "palette_summary": f"{primary} (60%) + {secondary} (30%) + {accent} (10%)",
        "recommendations": recommendations,
        "source": "Architectural interior guidelines from room specs, spatial geometry, and ergonomic rules",
    }
