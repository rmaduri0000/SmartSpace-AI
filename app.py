"""
SmartSpace AI - Main Web Application and REST API Server
Integrates YOLO Computer Vision with DQN Reinforcement Learning Layout Optimizer.
Provides the multi-page design flow:
- / (Landing Page with 3 Feature Cards & How It Works)
- /create-project (Project Name, Room Type, Description)
- /room-details (1. Room Info, 2. Dimensions in ft, 3. Budget in ₹, 4. Preferences, 5. Photo Upload)
- /studio (Interactive 2D & 3D WebGL Layout Optimization Studio)
"""
import os
import json
import io
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

# Keep optional TensorFlow startup quiet before importing the model layer.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from config import TARGET_CLASSES, CLASS_COLORS, FURNITURE_SPECS, SAMPLE_ROOMS
from vision.detector import VisionStateExtractor
from engine.interior_env import InteriorEnv
from engine.dqn_agent import DQNAgent

PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # 16 MB upload limit

# Global service singletons
vision_extractor = VisionStateExtractor()
dqn_agent = DQNAgent(state_dim=133, action_dim=48)

# In-memory session store for projects and rooms
PROJECTS_STORE = {}
ROOMS_STORE = {}

# =========================================================================
# Multi-Page HTML View Routes
# =========================================================================

@app.route("/")
def landing_page():
    """Renders the SmartSpace overview page."""
    return render_template("landing.html", page_key="home")

@app.route("/create-project")
def create_project_page():
    """Renders the first step of the room planning flow."""
    return render_template("create_project.html", page_key="project")

@app.route("/room-details")
def room_details_page():
    """Renders room specifications and upload step."""
    return render_template("room_details.html", page_key="room")

@app.route("/studio")
def studio_page():
    """Renders Interactive 2D/3D Studio with DQN Layout AI."""
    return render_template(
        "studio.html",
        classes=TARGET_CLASSES,
        colors=CLASS_COLORS,
        specs=FURNITURE_SPECS,
        sample_rooms=SAMPLE_ROOMS,
        page_key="studio"
    )

@app.route("/data/samples/<path:filename>")
def serve_sample_file(filename):
    return send_from_directory(PROJECT_ROOT / "data" / "samples", filename)

# =========================================================================
# Project & Room State REST APIs
# =========================================================================

@app.route("/api/projects", methods=["POST"])
def create_project_api():
    """Creates a new interior design project session."""
    data = request.get_json() or {}
    project_id = str(uuid.uuid4())[:8]
    project = {
        "id": project_id,
        "name": data.get("name", "My Interior Project"),
        "roomType": data.get("roomType", "Living Room"),
        "description": data.get("description", "")
    }
    PROJECTS_STORE[project_id] = project
    return jsonify({"success": True, "project": project})

@app.route("/api/rooms", methods=["POST"])
def save_room_api():
    """
    Saves room specifications from /room-details form.
    Converts dimensions from feet to meters (1 ft = 0.3048 m).
    Processes uploaded photo via YOLO if provided.
    """
    try:
        # Form field extractions
        room_name = request.form.get("roomName", "Master Room")
        room_type = request.form.get("roomType", "Master Bedroom")
        
        # Dimensions in feet converted to meters
        length_ft = float(request.form.get("length", 12.0))
        width_ft = float(request.form.get("width", 10.0))
        height_ft = float(request.form.get("height", 10.0))
        
        length_m = round(length_ft * 0.3048, 2)
        width_m = round(width_ft * 0.3048, 2)
        height_m = round(height_ft * 0.3048, 2)
        
        num_doors = int(request.form.get("doors", 1))
        num_windows = int(request.form.get("windows", 1))
        budget_inr = float(request.form.get("budget", 20000.0))
        
        design_style = request.form.get("designStyle", "Modern")
        primary_color = request.form.get("primaryColor", "White")
        secondary_color = request.form.get("secondaryColor", "Beige")
        material = request.form.get("preferredMaterial", "Wood")
        description = request.form.get("description", "")
        
        # Check uploaded room photo for YOLO vision analysis
        uploaded_file = request.files.get("roomPhoto")
        annotated_img_b64 = None
        detected_furniture = None
        vision_result = None
        
        if uploaded_file and uploaded_file.filename != "":
            img_bytes = uploaded_file.read()
            vision_result = vision_extractor.process_room_image(
                img_bytes, room_dimensions={"width": width_m, "length": length_m, "height": height_m}
            )
            annotated_img_b64 = vision_result.get("annotated_image")
            if "room_state" in vision_result:
                detected_furniture = vision_result["room_state"].get("initial_furniture")

        # Initial architectural elements
        door_info = {"wall": "south", "offset": max(0.4, width_m * 0.2), "width": 0.9}
        windows = []
        for w_idx in range(num_windows):
            windows.append({
                "wall": "north",
                "offset": max(0.4, (width_m / (num_windows + 1)) * (w_idx + 1) - 0.6),
                "width": 1.4
            })
            
        # If no furniture was detected, use a recommendation that matches the selected room.
        if not detected_furniture:
            if "bedroom" in room_type.lower():
                detected_furniture = [
                    {"id": "bed_1", "type": "bed", "x": width_m * 0.5, "y": length_m * 0.7, "width": 1.6, "depth": 2.0, "height": 0.8, "rotation": 0, "cost": 12000, "label": "Queen Bed"},
                    {"id": "wardrobe_1", "type": "wardrobe", "x": width_m * 0.8, "y": length_m * 0.3, "width": 1.5, "depth": 0.6, "height": 2.1, "rotation": 90, "cost": 8500, "label": "2-Door Wardrobe"},
                    {"id": "desk_1", "type": "desk", "x": width_m * 0.25, "y": length_m * 0.3, "width": 1.2, "depth": 0.6, "height": 0.75, "rotation": 0, "cost": 4200, "label": "Study Desk"},
                    {"id": "chair_1", "type": "chair", "x": width_m * 0.25, "y": length_m * 0.45, "width": 0.6, "depth": 0.6, "height": 0.85, "rotation": 0, "cost": 1500, "label": "Desk Chair"}
                ]
            elif "office" in room_type.lower():
                detected_furniture = [
                    {"id": "desk_1", "type": "desk", "x": width_m * 0.5, "y": length_m * 0.65, "rotation": 0, "cost": 4200, "label": "Study Desk"},
                    {"id": "chair_1", "type": "chair", "x": width_m * 0.5, "y": length_m * 0.4, "rotation": 0, "cost": 1500, "label": "Office Chair"},
                    {"id": "cabinet_1", "type": "cabinet", "x": width_m * 0.82, "y": length_m * 0.65, "rotation": 90, "cost": 3500, "label": "Storage Cabinet"}
                ]
            elif "dining" in room_type.lower():
                detected_furniture = [
                    {"id": "table_1", "type": "table", "x": width_m * 0.5, "y": length_m * 0.55, "rotation": 0, "cost": 3500, "label": "Dining Table"},
                    {"id": "chair_1", "type": "chair", "x": width_m * 0.5, "y": length_m * 0.22, "rotation": 0, "cost": 1500, "label": "Dining Chair"},
                    {"id": "chair_2", "type": "chair", "x": width_m * 0.5, "y": length_m * 0.84, "rotation": 180, "cost": 1500, "label": "Dining Chair"},
                    {"id": "cabinet_1", "type": "cabinet", "x": width_m * 0.82, "y": length_m * 0.55, "rotation": 90, "cost": 3500, "label": "Sideboard"}
                ]
            elif "kitchen" in room_type.lower():
                detected_furniture = [
                    {"id": "table_1", "type": "table", "x": width_m * 0.5, "y": length_m * 0.5, "rotation": 0, "cost": 3500, "label": "Kitchen Island"},
                    {"id": "cabinet_1", "type": "cabinet", "x": width_m * 0.82, "y": length_m * 0.5, "rotation": 90, "cost": 3500, "label": "Storage Cabinet"}
                ]
            elif "bathroom" in room_type.lower():
                detected_furniture = [
                    {"id": "cabinet_1", "type": "cabinet", "x": width_m * 0.75, "y": length_m * 0.55, "rotation": 90, "cost": 3500, "label": "Vanity Cabinet"}
                ]
            else:  # Living room / studio apartment
                detected_furniture = [
                    {"id": "sofa_1", "type": "sofa", "x": width_m * 0.5, "y": length_m * 0.3, "width": 2.1, "depth": 0.9, "height": 0.85, "rotation": 0, "cost": 14000, "label": "3-Seater Sofa"},
                    {"id": "table_1", "type": "table", "x": width_m * 0.5, "y": length_m * 0.55, "width": 1.2, "depth": 0.7, "height": 0.45, "rotation": 0, "cost": 3500, "label": "Coffee Table"},
                    {"id": "tv_1", "type": "tv", "x": width_m * 0.5, "y": length_m * 0.88, "width": 1.2, "depth": 0.15, "height": 0.7, "rotation": 180, "cost": 8000, "label": "55\" TV & Console"},
                    {"id": "chair_1", "type": "chair", "x": width_m * 0.2, "y": length_m * 0.55, "width": 0.7, "depth": 0.7, "height": 0.8, "rotation": 90, "cost": 2200, "label": "Lounge Chair"}
                ]
                
        for item in detected_furniture:
            specs = FURNITURE_SPECS.get(item.get("type", "chair"), FURNITURE_SPECS["chair"])
            item.setdefault("width", specs["width"])
            item.setdefault("depth", specs["depth"])
            item.setdefault("height", specs["height"])
            item.setdefault("cost", specs["base_cost"])
            item.setdefault("preferred_wall", specs.get("preferred_wall", False))

        room_id = str(uuid.uuid4())[:8]
        room_obj = {
            "id": room_id,
            "name": room_name,
            "room_type": room_type,
            "dimensions": {"width": width_m, "length": length_m, "height": height_m},
            "door": door_info,
            "windows": windows,
            "budget": budget_inr,
            "style": design_style,
            "primaryColor": primary_color,
            "secondaryColor": secondary_color,
            "material": material,
            "furniture": detected_furniture,
            "annotated_image": annotated_img_b64,
            "vision_model_available": vision_extractor.detector.available,
            "vision_model_backend": vision_extractor.detector.engine_type,
            "vision_detected_count": (vision_result or {}).get("detected_count", 0)
        }
        
        ROOMS_STORE[room_id] = room_obj
        return jsonify({"success": True, "room": room_obj})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================================================================
# AI Engine REST APIs (YOLO, DQN, Evaluation)
# =========================================================================

@app.route("/api/sample-rooms", methods=["GET"])
def get_sample_rooms():
    """Returns available pre-configured sample rooms."""
    return jsonify({
        "success": True,
        "rooms": SAMPLE_ROOMS
    })

@app.route("/api/detect", methods=["POST"])
def detect_room():
    """Runs YOLO 10-class interior object detection."""
    try:
        img_input = None
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                img_input = file.read()
        elif request.is_json:
            data = request.get_json()
            if "image_b64" in data:
                img_input = data["image_b64"]
            elif "sample_id" in data:
                sample_path = os.path.join("data", "samples", "sample_room.jpg")
                if os.path.exists(sample_path):
                    with open(sample_path, "rb") as f:
                        img_input = f.read()
                        
        if img_input is None:
            sample_path = os.path.join("data", "samples", "sample_room.jpg")
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    img_input = f.read()
            else:
                return jsonify({"success": False, "error": "No image provided"}), 400
                
        result = vision_extractor.process_room_image(img_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/optimize", methods=["POST"])
def optimize_layout():
    """Runs the Deep Q-Network (DQN) layout optimizer."""
    try:
        room_data = request.get_json() or {}
        max_steps = max(0, min(100, int(room_data.get("max_steps", 25))))
        
        env = InteriorEnv(room_data)
        trajectory = dqn_agent.optimize_layout_trajectory(env, max_steps=max_steps)
        final_layout = env.get_layout_dict()
        
        return jsonify({
            "success": True,
            "trajectory": trajectory,
            "final_layout": final_layout,
            "step_count": len(trajectory),
            "optimizer_trained": dqn_agent.is_trained,
            "optimizer_backend": dqn_agent.backend
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/evaluate", methods=["POST"])
def evaluate_custom_layout():
    """Evaluates real-time ergonomics, collisions, and A* circulation."""
    try:
        room_data = request.get_json() or {}
        env = InteriorEnv(room_data)
        metrics = env.evaluate_layout()
        return jsonify({"success": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/training-telemetry", methods=["GET"])
def get_training_telemetry():
    """Return only training curves accompanied by real installed checkpoints."""
    models_dir = PROJECT_ROOT / "data" / "models"
    dqn_file = models_dir / "dqn_training_history.json"
    yolo_file = PROJECT_ROOT / "runs" / "interior_yolo" / "yolo_training_telemetry.json"
    dqn_data = None
    yolo_data = None
    if dqn_agent.is_trained and dqn_file.is_file():
        try:
            candidate = json.loads(dqn_file.read_text(encoding="utf-8"))
            if candidate.get("real_training") is True:
                dqn_data = candidate
        except (OSError, ValueError):
            pass
    if vision_extractor.detector.available and yolo_file.is_file():
        try:
            candidate = json.loads(yolo_file.read_text(encoding="utf-8"))
            if candidate.get("real_training") is True:
                yolo_data = candidate
        except (OSError, ValueError):
            pass
    return jsonify({"success": True, "dqn": dqn_data, "yolo": yolo_data})


@app.route("/api/system-status", methods=["GET"])
def get_system_status():
    """Report the actual installed model state for the studio UI."""
    detector = vision_extractor.detector
    return jsonify({
        "success": True,
        "vision": {
            "available": detector.available,
            "backend": detector.engine_type,
            "model": detector.model_path.name if detector.available else None,
        },
        "optimizer": {
            "available": True,
            "trained": dqn_agent.is_trained,
            "backend": dqn_agent.backend,
            "mode": "trained DQN with spatial search" if dqn_agent.is_trained else "spatial search (checkpoint not trained)",
        },
    })

@app.route("/api/export-spec", methods=["POST"])
def export_spec():
    """Generates a downloadable design spec sheet."""
    try:
        data = request.get_json() or {}
        layout = data.get("layout", {})
        furniture = layout.get("furniture", [])
        metrics = layout.get("metrics", {})
        
        spec_text = []
        spec_text.append("=" * 60)
        spec_text.append("           SMARTSPACE AI · INTERIOR DESIGN SPECIFICATION")
        spec_text.append("=" * 60)
        spec_text.append(f"Room Dimensions : {layout.get('room_width', 4.8)}m x {layout.get('room_length', 4.0)}m")
        spec_text.append(f"Ergonomics Score: {metrics.get('ergonomics_score', 0)}%")
        spec_text.append(f"Circulation     : {metrics.get('circulation_ratio', 0)*100:.0f}% Reachable")
        spec_text.append(f"Collisions      : {metrics.get('collision_count', 0)} Detected")
        spec_text.append(f"Total Cost      : ₹{metrics.get('total_cost', 0):,}")
        spec_text.append(f"Budget Limit    : ₹{layout.get('budget', 0):,}")
        spec_text.append("-" * 60)
        spec_text.append("FURNITURE SCHEDULE:")
        for idx, item in enumerate(furniture, 1):
            spec_text.append(
                f"  {idx}. {item.get('label', item['type']).upper():<24} | "
                f"Pos: ({item['x']:.2f}m, {item['y']:.2f}m) | "
                f"Rot: {item.get('rotation', 0):>3} deg | "
                f"Size: {item['width']:.2f}x{item['depth']:.2f}m | "
                f"₹{item.get('cost', 0):,}"
            )
        spec_text.append("=" * 60)
        spec_text.append("Generated with SmartSpace AI · Interior Design Studio")
        
        buffer = io.BytesIO()
        buffer.write("\n".join(spec_text).encode("utf-8"))
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name="smartspace_design_specification.txt",
            mimetype="text/plain"
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(_error):
    return render_template("404.html", page_key="not-found"), 404


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  [*] Starting SmartSpace AI Platform")
    print("  [*] Landing Page:   http://127.0.0.1:5000/")
    print("  [*] Create Project: http://127.0.0.1:5000/create-project")
    print("  [*] Room Details:   http://127.0.0.1:5000/room-details")
    print("  [*] AI Studio:      http://127.0.0.1:5000/studio")
    print("="*65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
