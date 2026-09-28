"""YOLO inference for interior objects using Ultralytics or ONNX Runtime."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import TARGET_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class YOLOInteriorDetector:
    """Loads real model weights and returns detections in original-image space."""

    def __init__(self, model_path: Optional[str] = None):
        requested = Path(model_path) if model_path else Path("data/models/yolo_interior.pt")
        self.model_path = requested if requested.is_absolute() else PROJECT_ROOT / requested
        self.classes = TARGET_CLASSES
        self.loaded_model = None
        self.engine_type = "unavailable"
        self.available = False
        self.input_size = (640, 640)
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Load the first usable supported checkpoint; absence is an explicit state."""
        candidates = [self.model_path]
        onnx_path = self.model_path.with_suffix(".onnx")
        if onnx_path not in candidates:
            candidates.append(onnx_path)

        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix.lower() == ".pt":
                try:
                    from ultralytics import YOLO
                    self.loaded_model = YOLO(str(path))
                    self.engine_type = "ultralytics"
                    self.available = True
                    self.model_path = path
                    print(f"[YOLO] Loaded weights: {path.name}")
                    return
                except Exception as error:
                    print(f"[YOLO] Could not load {path.name}: {error}")
            elif path.suffix.lower() == ".onnx":
                try:
                    import onnxruntime as ort
                    self.loaded_model = ort.InferenceSession(
                        str(path), providers=ort.get_available_providers()
                    )
                    input_shape = self.loaded_model.get_inputs()[0].shape
                    if len(input_shape) == 4:
                        height, width = input_shape[-2:]
                        self.input_size = (
                            int(width) if isinstance(width, int) else 640,
                            int(height) if isinstance(height, int) else 640,
                        )
                    self.engine_type = "onnx"
                    self.available = True
                    self.model_path = path
                    print(f"[YOLO] Loaded ONNX weights: {path.name}")
                    return
                except Exception as error:
                    print(f"[YOLO] Could not load {path.name}: {error}")

        print(f"[YOLO] No supported weights found at {self.model_path} or {onnx_path}")

    def predict(self, image: Image.Image, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Run detection. Returns an empty list when no real model is available."""
        if not self.available or self.loaded_model is None:
            return []
        image = image.convert("RGB")
        if self.engine_type == "ultralytics":
            results = self.loaded_model(image, conf=conf_threshold, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(self._format_detection(cls_id, confidence, xyxy, image.size))
            return detections
        if self.engine_type == "onnx":
            return self._predict_onnx(image, conf_threshold)
        return []

    def _predict_onnx(self, image: Image.Image, conf_threshold: float) -> List[Dict[str, Any]]:
        input_w, input_h = self.input_size
        original_w, original_h = image.size
        scale = min(input_w / original_w, input_h / original_h)
        resized_w, resized_h = int(round(original_w * scale)), int(round(original_h * scale))
        resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (input_w, input_h), (114, 114, 114))
        pad_x, pad_y = (input_w - resized_w) // 2, (input_h - resized_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        tensor = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        input_meta = self.loaded_model.get_inputs()[0]
        raw_outputs = self.loaded_model.run(None, {input_meta.name: tensor})
        if not raw_outputs:
            return []
        rows = self._output_rows(np.asarray(raw_outputs[0]))
        if rows.size == 0:
            return []

        boxes: List[List[float]] = []
        scores: List[float] = []
        class_ids: List[int] = []
        for row in rows:
            if row.size == 6:  # Exported with NMS: x1, y1, x2, y2, confidence, class.
                x1, y1, x2, y2, confidence, class_id = row.tolist()
                cls_id = int(class_id)
            else:
                # Standard YOLOv8/11 output has cx, cy, w, h and class scores;
                # YOLOv5 also includes an objectness column.
                if row.size < 5:
                    continue
                cx, cy, width, height = row[:4]
                class_scores = row[4:]
                if class_scores.size == len(self.classes) + 1:
                    objectness = float(class_scores[0])
                    class_scores = class_scores[1:] * objectness
                cls_id = int(np.argmax(class_scores))
                confidence = float(class_scores[cls_id])
                x1, y1 = cx - width / 2, cy - height / 2
                x2, y2 = cx + width / 2, cy + height / 2
            if cls_id < 0 or cls_id >= len(self.classes) or confidence < conf_threshold:
                continue

            # Some exports return normalized coordinates instead of model pixels.
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                x1, x2 = x1 * input_w, x2 * input_w
                y1, y2 = y1 * input_h, y2 * input_h
            x1 = max(0.0, min(original_w, (x1 - pad_x) / scale))
            x2 = max(0.0, min(original_w, (x2 - pad_x) / scale))
            y1 = max(0.0, min(original_h, (y1 - pad_y) / scale))
            y2 = max(0.0, min(original_h, (y2 - pad_y) / scale))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(cls_id)

        keep = self._nms(boxes, scores, class_ids, iou_threshold=0.45)
        return [self._format_detection(class_ids[i], scores[i], boxes[i], image.size) for i in keep]

    @staticmethod
    def _output_rows(output: np.ndarray) -> np.ndarray:
        output = np.squeeze(output)
        if output.ndim == 1:
            output = output[None, :]
        if output.ndim != 2:
            return np.empty((0, 0), dtype=np.float32)
        # YOLO exports commonly use [features, candidates]. Transpose that form.
        if output.shape[0] in (6, 14, 15) and output.shape[1] > output.shape[0]:
            output = output.T
        return output.astype(np.float32, copy=False)

    @staticmethod
    def _nms(boxes: List[List[float]], scores: List[float], class_ids: List[int],
             iou_threshold: float) -> List[int]:
        kept: List[int] = []
        for cls_id in set(class_ids):
            order = sorted((i for i, cls in enumerate(class_ids) if cls == cls_id),
                           key=lambda i: scores[i], reverse=True)
            while order:
                current = order.pop(0)
                kept.append(current)
                x1, y1, x2, y2 = boxes[current]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                remaining = []
                for idx in order:
                    ox1, oy1, ox2, oy2 = boxes[idx]
                    iw = max(0.0, min(x2, ox2) - max(x1, ox1))
                    ih = max(0.0, min(y2, oy2) - max(y1, oy1))
                    intersection = iw * ih
                    other_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
                    union = area + other_area - intersection
                    if union <= 0 or intersection / union <= iou_threshold:
                        remaining.append(idx)
                order = remaining
        return sorted(kept, key=lambda i: scores[i], reverse=True)

    def _format_detection(self, cls_id: int, confidence: float, xyxy: List[float],
                          image_size: Tuple[int, int]) -> Dict[str, Any]:
        img_w, img_h = image_size
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        width, height = (x2 - x1) / img_w, (y2 - y1) / img_h
        return {
            "class_id": cls_id,
            "class_name": self.classes[cls_id],
            "confidence": round(float(confidence), 3),
            "bbox_xyxy": [round(float(value), 1) for value in xyxy],
            "bbox_norm": [round(float(value), 4) for value in (cx, cy, width, height)],
        }
