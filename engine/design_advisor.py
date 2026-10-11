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


ROOM_TAXONOMY = {
    "living_room": {
        "name": "Living Room",
        "furniture": [
            "Sofa set or modular sectional sofa",
            "Coffee table with rounded or soft edges",
            "Matching or complementary side tables",
            "Accent armchair or swivel reading chair (optional)",
            "Display unit or curio cabinet (optional)"
        ],
        "appliances": [
            "Wall-mounted smart television (55\"–65\")",
            "Integrated media console with soundbar / home theater chase"
        ],
        "lighting": [
            "Dimmable 2700K ambient ceiling downlights",
            "Sculptural floor lamp by the seating nook",
            "Indirect warm LED bias backlighting behind the TV console"
        ],
        "decorations": [
            "Floor-to-ceiling curtains or tailored roman blinds",
            "Textured area rug anchoring the front furniture legs",
            "Coordinated scatter cushions and knit throws",
            "Curated canvas wall art and paintings",
            "Decorative architectural wall mirror",
            "Air-purifying indoor plants (Fiddle-leaf fig, Monstera)"
        ],
        "storage_units": [
            "Low-profile media console with concealed cable management",
            "Floating wall shelves or integrated display cabinet"
        ],
        "materials": [
            "Natural oak or warm walnut timber",
            "Tactile bouclé, textured linen, or top-grain leather upholstery",
            "Matte black or brushed brass hardware accents"
        ],
        "color_schemes": {
            "dominant_60": "Warm ivory or alabaster white (walls & ceiling)",
            "secondary_30": "Earthy taupe, greige, or soft charcoal (upholstery & rugs)",
            "accent_10": "Champagne brass, terracotta, or forest green (cushions & metal accents)"
        },
        "layout": "Maintain 45 cm (18\") clearance between coffee table and sofa for comfortable legroom. Position TV console at seated eye level (105 cm) with 1.8–2.4 m viewing distance. Preserve an open 80–90 cm circulation spine from the doorway.",
        "essential_items": [
            "Sofa set / Sectional sofa",
            "Coffee table",
            "TV unit and television",
            "Curtains or blinds",
            "Ceiling ambient lighting and area rug"
        ],
        "optional_accessories": [
            "Accent armchair",
            "Side tables",
            "Display curio unit",
            "Sculptural floor lamp",
            "Decorative mirror and indoor plants"
        ],
        "budget_alternatives": {
            "budget_friendly": "Flat-pack laminate TV unit, durable stain-resistant polyester-blend 3-seater sofa, machine-washable printed area rug",
            "premium": "Custom solid white oak media console, Italian top-grain leather or bouclé modular sectional, hand-knotted wool-silk area rug"
        }
    },
    "bedroom": {
        "name": "Bedroom",
        "furniture": [
            "Queen or double bed with ergonomic mattress and padded headboard",
            "Bedside tables with drawers (1 or 2)",
            "Wardrobe or built-in closet",
            "Dressing table with mirror",
            "Study table or compact work desk (if required)"
        ],
        "appliances": [
            "Silent whisper inverter split AC",
            "Bedside reading lamps or focused wall sconces"
        ],
        "lighting": [
            "Warm 2700K ambient ceiling lighting with dimmer control",
            "Independent bedside reading sconces mounted at 105 cm height"
        ],
        "decorations": [
            "Dual-track curtains (sheers for filtered daylight + blackout drapery for sleep)",
            "Plush wool bedside rug extending 65 cm around mattress",
            "Layered pillows, scatter cushions, and quilted bedding throw",
            "Framed minimalist wall art and decorative tabletop accents"
        ],
        "storage_units": [
            "Full-height 2-to-3 door wardrobe with hanging rail and shelving",
            "Bedside storage drawers for night essentials",
            "Under-bed hydraulic or pull-out storage bins"
        ],
        "materials": [
            "Light birch, blonde pine, or natural oak",
            "Breathable washed linen and cotton percale bedding",
            "Brushed nickel, matte chrome, or matte black hardware"
        ],
        "color_schemes": {
            "dominant_60": "Soft oatmeal, warm beige, or off-white",
            "secondary_30": "Calming sage green, dusty blue, or cashmere taupe",
            "accent_10": "Brushed brass, burnt amber, or slate graphite"
        },
        "layout": "Position bed in the command position against a solid wall facing the entrance without direct draft alignment. Preserve 65–75 cm walking clearance on both sides of the mattress.",
        "essential_items": [
            "Bed with mattress and headboard",
            "At least one bedside table",
            "Wardrobe or closet",
            "Curtains or blackout blinds",
            "Ambient ceiling light and bedside reading lamp"
        ],
        "optional_accessories": [
            "Dressing table with mirror",
            "Compact study desk and chair",
            "Accent bedside rug",
            "Decorative wall art",
            "Secondary bedside table"
        ],
        "budget_alternatives": {
            "budget_friendly": "Upholstered platform bed with engineered wood frame, ready-to-assemble 2-door wardrobe, cotton linen blends",
            "premium": "Solid oak or teak bedstead with custom fluted headboard, floor-to-ceiling built-in wardrobe with glass tinted doors, pure Belgian flax linen"
        }
    },
    "master_bedroom": {
        "name": "Master Bedroom",
        "furniture": [
            "King-size bed with orthopedic mattress and statement headboard",
            "Two symmetric bedside tables with soft-close drawers",
            "Large master wardrobe or walk-in closet",
            "Dressing table with full-length mirror",
            "Lounge chair or cozy bedroom seating nook",
            "TV unit and media console (if required)"
        ],
        "appliances": [
            "55\" wall-mounted 4K smart television with concealed cable conduit",
            "High-efficiency inverter split AC with whisper-quiet sleep mode",
            "Motorized motorized blackout blinds (optional)"
        ],
        "lighting": [
            "3-zone architectural lighting scheme: dimmable perimeter cove ambient lighting",
            "Dedicated high-CRI wardrobe/dressing task lighting",
            "Hotel-style bedside pendant lights and directional reading spots"
        ],
        "decorations": [
            "8x10 ft hand-tufted textured wool rug under the King bed",
            "Premium 500-thread-count cotton sateen bedding with velvet cushions",
            "Floor-to-ceiling pinch-pleat blackout drapery",
            "Architectural full-length mirror and oversized canvas art"
        ],
        "storage_units": [
            "Large wardrobe or walk-in closet with integrated valet rods and LED channels",
            "Twin bedside drawer units",
            "Dressing vanity with velvet-lined jewelry and cosmetics organizers"
        ],
        "materials": [
            "Wire-brushed fumed oak or American walnut",
            "Fluted architectural wall panels",
            "Plush velvet and textured wool",
            "Brushed champagne bronze or graphite metal accents"
        ],
        "color_schemes": {
            "dominant_60": "Warm ivory, French vanilla, or cashmere grey",
            "secondary_30": "Slate taupe, rich greige, or deep navy",
            "accent_10": "Champagne gold, brushed brass, or emerald green"
        },
        "layout": "Center the King bed on the primary feature wall with symmetric 60 cm approach corridors. Provide 90 cm clear swing depth in front of wardrobes and maintain an open 100 cm seating sanctuary.",
        "essential_items": [
            "King-size bed with premium mattress & headboard",
            "Two bedside tables",
            "Large wardrobe or walk-in closet",
            "Layered ambient and bedside lighting",
            "Blackout curtains or motorized blinds"
        ],
        "optional_accessories": [
            "Lounge armchair with footstool",
            "Dressing table with full-length LED mirror",
            "Wall-mounted TV unit",
            "Bench at foot of bed",
            "Architectural cove lighting"
        ],
        "budget_alternatives": {
            "budget_friendly": "Queen storage bed with channel tufted headboard, modular sliding wardrobe, dimmable LED ring pendants",
            "premium": "Custom King platform bed in fumed oak, walk-in glass-door wardrobe with auto-sensor lighting, designer lounge chair in bouclé"
        }
    },
    "kitchen": {
        "name": "Kitchen",
        "furniture": [
            "Modular base cabinets with heavy-duty tandem drawers",
            "Overhead wall cabinets with lift-up hydraulic shutters",
            "Continuous stone countertop and full-height backsplash",
            "Kitchen prep island or dining breakfast counter (space permitting)"
        ],
        "appliances": [
            "Built-in 3 or 4-burner gas stove or induction cooktop",
            "High-suction kitchen chimney hood or ducted exhaust system",
            "Double-door frost-free refrigerator",
            "Built-in microwave or convection oven tower",
            "Dishwasher (if required / space permitting)"
        ],
        "fixtures": [
            "Stainless steel double-bowl sink or granite composite basin with swan-neck mixer faucet",
            "Stain-resistant quartz or sintered stone countertops",
            "Heat-resistant ceramic or porcelain backsplash tiles"
        ],
        "lighting": [
            "3500K–4000K high-CRI LED task strip lighting directly under upper cabinets",
            "Glare-free recessed ceiling downlights",
            "Warm pendant lights centered over kitchen island or breakfast bar"
        ],
        "decorations": [
            "Ceramic herb planters on window sill",
            "Magnetic knife rack or artisan wooden spice block",
            "Countertop fruit bowl or marble organizer tray"
        ],
        "storage_units": [
            "Dedicated cutlery & spoon drawer unit with partitioned organizer trays",
            "Daily essentials overhead wall cupboard for spices, cups, and bowls",
            "Tall grocery pantry storage cabinet with pull-out wire baskets",
            "Under-sink plumbing and cleaning supply cabinet"
        ],
        "materials": [
            "Scratch- and stain-resistant engineered quartz or porcelain surfaces",
            "Marine-grade plywood shutters with acrylic or anti-fingerprint laminate",
            "304-grade stainless steel fixtures and brushed brass pulls"
        ],
        "color_schemes": {
            "dominant_60": "Alabaster white or light dove grey (cabinets & ceiling)",
            "secondary_30": "Natural warm oak, sage green, or charcoal grey (island & accents)",
            "accent_10": "Matte black, brushed nickel, or warm brass (hardware & taps)"
        },
        "layout": "Strictly honor the Kitchen Work Triangle: ensure total perimeter between sink, stove, and refrigerator is between 4.0 and 7.5 m (each leg 1.2–2.7 m). Position cutlery and spoon drawers directly beside the prep counter, mount daily essentials cupboards overhead within arm's reach, and maintain 105–120 cm clearance around the prep island.",
        "essential_items": [
            "Modular base cabinets & countertop",
            "Kitchen wash basin sink and mixer faucet",
            "Gas stove or induction cooktop",
            "Kitchen chimney or exhaust system",
            "Refrigerator",
            "Daily essentials overhead wall cupboard",
            "Cutlery & spoon drawer unit"
        ],
        "optional_accessories": [
            "Kitchen prep island",
            "Dishwasher",
            "Built-in microwave tower",
            "Tall pantry pullout storage unit",
            "Under-cabinet LED task illumination"
        ],
        "budget_alternatives": {
            "budget_friendly": "High-pressure laminate shutters, granite countertop, standalone gas stove, stainless steel sink",
            "premium": "Seamless Calacatta quartz countertops with waterfall island, Blum servo-drive motorized drawers, integrated touch induction hob & concealed dishwasher"
        }
    },
    "bathroom": {
        "name": "Bathroom",
        "furniture": [
            "Washbasin with floating or freestanding vanity cabinet",
            "Mirror or LED-illuminated mirrored medicine cabinet",
            "Waterproof wall shelves or tall linen storage cabinet"
        ],
        "appliances": [
            "High-velocity low-noise moisture exhaust fan",
            "Instant or storage water geyser (concealed)"
        ],
        "fixtures": [
            "Rimless toilet with soft-close seat and dual-flush cistern",
            "Walk-in shower area or freestanding soaking bathtub (space permitting)",
            "Frameless tempered glass shower enclosure or partition",
            "Thermostatic rain showerhead and diverter mixer faucets",
            "Stainless steel towel racks and robe hooks"
        ],
        "lighting": [
            "Vertical eye-level vanity sconces (IP44 rated at ~165 cm) to eliminate unflattering face shadows",
            "Warm indirect LED cove lighting for night safety",
            "Dedicated ceiling-recessed IP65 waterproof shower downlight"
        ],
        "decorations": [
            "Matte ceramic soap dispensers and toothbrush tumblers",
            "Plush non-slip waffle-weave bath mats",
            "Privacy frosted glass awning window",
            "Minimalist moisture-tolerant succulent or fern"
        ],
        "flooring_and_walls": [
            "Anti-slip R10/R11 water-resistant porcelain floor tiles",
            "Floor-to-ceiling waterproof ceramic or marble-look wall tiles"
        ],
        "storage_units": [
            "Under-basin vanity drawers with plumbing cutouts",
            "Recessed wall niche in shower for shampoo and soaps",
            "Mirrored cabinet with internal electric shaver socket"
        ],
        "materials": [
            "Moisture-resistant marine ply with waterproof laminate",
            "Sintered stone or Corian seamless basin counter",
            "10mm toughened safety glass",
            "PVD-coated matte black, brushed brass, or chrome fixtures"
        ],
        "color_schemes": {
            "dominant_60": "Warm off-white, light grey, or bone travertine (wall tiles)",
            "secondary_30": "Natural slate, terrazzo, or ribbed timber (vanity & feature wall)",
            "accent_10": "Matte black, brushed champagne gold, or chrome (fixtures & hardware)"
        },
        "layout": "Maintain 75 cm clear zone in front of vanity basin and toilet. Set vanity counter height at 85–90 cm with mirror bottom 15–20 cm above faucet. Separate wet shower zone with glass partition and a 1:50 floor slope towards channel drain.",
        "essential_items": [
            "Toilet and flush system",
            "Washbasin with vanity cabinet",
            "Mirror or mirrored cabinet",
            "Showerhead and faucets",
            "Towel rack / hooks",
            "Anti-slip floor and waterproof wall tiles",
            "High-velocity exhaust fan"
        ],
        "optional_accessories": [
            "Frameless glass shower partition",
            "Freestanding soaking bathtub",
            "LED backlit anti-fog mirror",
            "Recessed shampoo niche",
            "Electric heated towel warmer"
        ],
        "budget_alternatives": {
            "budget_friendly": "Ceramic pedestal basin or single-door vanity, chrome mixer, framed shower curtain or fixed partial glass screen",
            "premium": "Floating quartz-top dual vanity, thermostatic concealed diverter, freestanding composite stone bathtub, heated anti-fog smart mirror"
        }
    },
    "dining_room": {
        "name": "Dining Room",
        "furniture": [
            "Dining table (sized for 2, 4, 6, or 8 persons based on spatial dimensions)",
            "Ergonomic matching dining chairs",
            "Crockery cabinet or low sideboard console",
            "Optional mobile serving trolley or buffet cabinet"
        ],
        "lighting": [
            "Sculptural statement pendant light or chandelier centered 75–85 cm (30–34\") above table top",
            "Dimmable warm perimeter wall sconces (2700K) on separate switches"
        ],
        "decorations": [
            "Woven linen table runner and ceramic floral centerpiece",
            "Curated wall art or large architectural framed mirror to enhance visual depth",
            "Floor-length sheer linen curtains or timber blinds",
            "Ambient decorative table lamp on sideboard"
        ],
        "storage_units": [
            "Sideboard console with felt-lined cutlery drawers and glassware shelves",
            "Open display shelving or glass-front crockery vitrine"
        ],
        "materials": [
            "Solid oak, teak, or American walnut tabletop",
            "Upholstered performance fabric or bentwood dining chairs",
            "Low-pile wool or flatweave rug",
            "Brushed brass or antique bronze hardware"
        ],
        "color_schemes": {
            "dominant_60": "Warm linen, oatmeal, or rich cream",
            "secondary_30": "Natural walnut wood, warm caramel, or soft charcoal",
            "accent_10": "Muted olive green, terracotta, or brushed champagne brass"
        },
        "layout": "Maintain minimum 90 cm (36\") clearance between table edge and surrounding walls or sideboards so guests can pull chairs out comfortably and circulate behind seated diners. Sizing guide: 2-seater (80x80 cm), 4-seater (120x80 cm), 6-seater (180x90 cm), 8-seater (240x100 cm).",
        "essential_items": [
            "Dining table suitable for room size",
            "Matching dining chairs",
            "Overhead pendant light or chandelier",
            "Curtains or blinds"
        ],
        "optional_accessories": [
            "Sideboard / Crockery cabinet",
            "Area rug extending 65 cm past table edge",
            "Serving trolley / bar cart",
            "Decorative table centerpiece",
            "Large accent mirror"
        ],
        "budget_alternatives": {
            "budget_friendly": "Engineered oak veneer table with molded polypropylene/wood leg chairs, modern geometric multi-bulb pendant",
            "premium": "Live-edge solid walnut table with custom trestle base, upholstered Italian leather dining chairs, designer hand-blown glass chandelier"
        }
    },
    "home_office": {
        "name": "Home Office",
        "furniture": [
            "Work desk or computer table (standard 140x70 cm or motorized electric standing desk)",
            "High-performance ergonomic task chair (lumbar support, 3D armrests)",
            "Optional visitor discussion chair or compact meeting table (space permitting)"
        ],
        "appliances": [
            "Computer or laptop workstation with docking hub",
            "External high-resolution monitor (single 32\" or dual 27\") mounted on monitor arm",
            "Ergonomic mechanical keyboard and wireless mouse"
        ],
        "lighting": [
            "High-CRI (90+) 4000K daylight task lamp with adjustable swing arm (500+ lux on desk)",
            "Anti-glare ceiling illumination",
            "Indirect bias lighting behind monitor to eliminate eye fatigue during late work"
        ],
        "storage_units": [
            "Modular bookshelf or tall display cabinet for books and reference material",
            "Lockable under-desk mobile pedestal with stationery and file drawers",
            "Concealed cable management conduit raceway and spine under desk"
        ],
        "decorations": [
            "Floating wall shelves with minimal architectural books and ceramic objects",
            "Low-pile sound-dampening acoustic rug",
            "Air-purifying indoor plants (Snake plant, Peace lily, Pothos)",
            "Glare-control roller blinds or sheer curtains"
        ],
        "materials": [
            "Solid birch, walnut, or laminate worktop",
            "Powder-coated heavy-gauge steel framing",
            "Breathable elastomeric mesh",
            "Acoustic PET felt panels"
        ],
        "color_schemes": {
            "dominant_60": "Clean soft white or light cement grey",
            "secondary_30": "Deep slate blue, dark walnut, or graphite",
            "accent_10": "Forest green, burnt orange, or matte black"
        },
        "layout": "Position work desk perpendicular to natural window light to eliminate direct monitor glare and front-light video conference calls. Ensure 85–100 cm space behind desk for task chair rollout. Maintain the 90-90-90 ergonomic posture.",
        "essential_items": [
            "Work desk / computer table",
            "Ergonomic task chair",
            "Computer workstation & monitor",
            "Task lighting lamp",
            "Cable management system",
            "Window blinds for glare control"
        ],
        "optional_accessories": [
            "Bookshelf unit / credenza",
            "Dual monitor articulation arm",
            "Visitor discussion chair",
            "Acoustic wall felt tiles",
            "Indoor plants & desk organizers"
        ],
        "budget_alternatives": {
            "budget_friendly": "Fixed-height study table with cable grommet, ergonomic mid-back mesh chair, clamp-on LED task light",
            "premium": "Dual-motor hardwood standing desk with memory presets, Herman Miller Aeron/Steelcase Gesture ergonomic chair, custom floor-to-ceiling library wall"
        }
    },
    "studio_apartment": {
        "name": "Studio Apartment",
        "furniture": [
            "Space-saving bed with hydraulic under-bed storage or convertible sofa bed",
            "Compact 2-seater sofa with slim arms",
            "Nesting coffee table with slide-under stools",
            "Multipurpose dining table or extendable breakfast counter",
            "Study desk or wall-mounted foldable workstation"
        ],
        "appliances": [
            "Compact modular kitchenette with 2-burner induction cooktop, slim chimney, built-in microwave, and under-counter refrigerator",
            "50-55\" wall-mounted smart television on articulated swivel arm"
        ],
        "lighting": [
            "Zoned ambient and task lighting with multi-circuit switches to illuminate sleeping, living, and cooking zones independently",
            "Slender floor lamp and under-cabinet strip LEDs"
        ],
        "storage_units": [
            "Floor-to-ceiling 2-door wardrobe with sliding mirror doors",
            "Double-sided open bookshelf serving as visual room partition",
            "Wall-mounted vertical floating storage cubbies"
        ],
        "decorations": [
            "Full-length architectural mirror positioned across from window to reflect daylight and visually double apartment dimensions",
            "Translucent linen ceiling-track curtains to soft-partition sleeping zone",
            "Zoned accent rugs defining living and sleeping micro-spaces",
            "Hanging or vertical indoor planters"
        ],
        "materials": [
            "Light reflective surfaces (lacquered white, light ash wood)",
            "Tempered glass and slim powder-coated steel",
            "Textured neutral upholstery with stain-guard treatment"
        ],
        "color_schemes": {
            "dominant_60": "Bright architectural white or warm ecru (maximizes light)",
            "secondary_30": "Scandi blonde timber, light greige, or soft stone",
            "accent_10": "Soft sage green, subtle blush, or matte black accents"
        },
        "layout": "Adopt smart open-plan micro-zoning: maintain an unobstructed 75–85 cm central circulation corridor connecting entry, kitchenette, bathroom, and bed. Choose dual-purpose modular furniture and elevate storage to the full wall height to keep floor planes visible.",
        "essential_items": [
            "Bed / Convertible sofa bed",
            "Compact wardrobe & under-bed storage",
            "Compact modular kitchenette with appliances",
            "Multipurpose dining/work surface",
            "Zoned lighting",
            "Window blinds"
        ],
        "optional_accessories": [
            "Open bookshelf room divider",
            "Full-length space-enhancing mirror",
            "Compact 2-seater sofa & nesting coffee table",
            "Swivel TV mount",
            "Translucent ceiling partition curtain"
        ],
        "budget_alternatives": {
            "budget_friendly": "Convertible 3-seater sofa bed with storage chaise, drop-leaf dining/work table, lightweight clothing rack & modular cubes",
            "premium": "Custom Italian hydraulic Murphy bed with integrated sofa and fold-down desk, full-height custom millwork kitchenette, acoustic room-divider slatted screen"
        }
    }
}


def generate_room_design_brief(
    room_type: str,
    dimensions: Any = None,
    style: str = "Modern",
    budget: float = 85000.0,
    primary_color: str = "",
    secondary_color: str = "",
    material_preference: str = "",
) -> Dict[str, Any]:
    """Return a comprehensive interior design brief compliant with design generation rules."""
    key = _room_key(room_type)
    taxonomy = ROOM_TAXONOMY.get(key, ROOM_TAXONOMY["living_room"])
    style_clean = str(style or "Modern").strip().title()
    style_key = style_clean.lower()
    accent = STYLE_ACCENTS.get(style_key, STYLE_ACCENTS.get("modern", "brushed brass and matte black"))
    wood = STYLE_WOODS.get(style_key, STYLE_WOODS.get("modern", "natural oak or walnut"))

    primary = primary_color.strip() if primary_color else taxonomy["color_schemes"]["dominant_60"].split("(")[0].strip()
    secondary = secondary_color.strip() if secondary_color else taxonomy["color_schemes"]["secondary_30"].split("(")[0].strip()
    material = material_preference.strip() if material_preference else wood

    # Dimension and area sizing
    width_m = 4.5
    length_m = 4.0
    if isinstance(dimensions, dict):
        width_m = float(dimensions.get("width", dimensions.get("width_m", 4.5)) or 4.5)
        length_m = float(dimensions.get("length", dimensions.get("length_m", 4.0)) or 4.0)
    area_sqm = round(width_m * length_m, 1)

    # Scale-aware dining capacity
    adjusted_furniture = list(taxonomy["furniture"])
    if key == "dining_room":
        if area_sqm < 12.0:
            adjusted_furniture[0] = "Dining table (compact 4-seater, 120x80 cm)"
        elif area_sqm < 20.0:
            adjusted_furniture[0] = "Dining table (generous 6-seater, 180x90 cm)"
        else:
            adjusted_furniture[0] = "Dining table (luxurious 8-seater banquet, 240x100 cm)"

    # AI Visualization Prompt generator
    ai_prompt = (
        f"Professional architectural interior photography of a {style_clean} {taxonomy['name']}, "
        f"{width_m:.1f}x{length_m:.1f} meters, styled with {', '.join(adjusted_furniture[:4])}. "
        f"Palette featuring {primary} walls, {secondary} textiles, and {accent}. "
        f"Crafted from {material} with natural daylight filtering through large windows. "
        f"Layered ambient lighting, clutter-free ergonomic circulation, 8k resolution, photorealistic, Unreal Engine 5 render style."
    )

    return {
        "room_type": key,
        "display_name": taxonomy["name"],
        "style": style_clean,
        "area_sqm": area_sqm,
        "budget_inr": budget,
        "furniture": adjusted_furniture,
        "appliances": taxonomy.get("appliances", []),
        "fixtures": taxonomy.get("fixtures", []),
        "lighting": taxonomy["lighting"],
        "decorations": taxonomy["decorations"],
        "storage_units": taxonomy["storage_units"],
        "materials": taxonomy["materials"],
        "color_palette": {
            "dominant_60": f"{primary} (60% walls & ceilings)",
            "secondary_30": f"{secondary} (30% large furniture & textiles)",
            "accent_10": f"{accent} (10% hardware & styling accents)",
            "primary": primary,
            "secondary": secondary,
            "accent": accent,
        },
        "layout": taxonomy["layout"],
        "essential_items": taxonomy["essential_items"],
        "optional_accessories": taxonomy["optional_accessories"],
        "budget_alternatives": taxonomy["budget_alternatives"],
        "ai_visualization_prompt": ai_prompt,
    }


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

    dimensions = room.get("dimensions")
    budget = float(room.get("budget", metrics.get("budget", 0)) or 0)
    design_brief = generate_room_design_brief(
        room_type=room_key,
        dimensions=dimensions,
        style=style,
        budget=budget,
        primary_color=primary,
        secondary_color=secondary,
        material_preference=material,
    )

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
        "design_brief": design_brief,
        "source": "Architectural interior guidelines from room specs, spatial geometry, and ergonomic rules",
    }
