# Step 4 repair

Status: FINALIZED

The wizard must advance from Step 3 without submitting; retain valid photos when navigating between steps; show and select recommendations; submit the selected room inputs and optional photo; recover from missing or failing YOLO; release pending/loading state on cancellation and browser Back; and hand the prepared 133-value state to Studio. Build the served frontend and verify actual backend responses and browser behavior.

## Photo inference implementation

Automatically obtain the official yolov8n checkpoint when no custom local PT/ONNX checkpoint is available; serialize initialization and shared inference; map model labels into the ten-class SmartSpace taxonomy; decode bounded JPG/PNG uploads with EXIF orientation; encode photo-derived furniture through the existing 133-value encoder; preserve room inputs, history, cancellation and honest fallback metadata. Wire Step 4 to the photo-room endpoint and retain detections in photo recommendation responses. Install the optional vision runtime into the active local environment and rebuild frontend artifacts.

## Compact furniture panel

Replace the Studio inventory DOM renderer with a React/Tailwind component mounted in the existing panel. Use a one/two/three-column compact grid, p-3 cards, grouped labels/dimensions and rotate/swap/delete controls. Outer panel hides overflow; the bounded inner grid is the only scrolling region. Preserve photo display, catalog, selection, layout redraws and metrics. Swap must replace the selected item through the existing catalog. Rebuild the served frontend bundle.
