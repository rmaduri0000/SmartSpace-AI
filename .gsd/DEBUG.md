# Step 4 wizard repair — 2026-09-30

Status: Resolved for the reproduced wizard flow.

## Findings

- The photo belonged to an unmounted Step 4 input rather than wizard state, so Back/Continue discarded it.
- Recommendation cards displayed bundle metrics but offered no selection or selected bundle submission.
- React reused the same footer button for Continue and launch, exposing the launch handler during rapid Step 3 interactions.
- The current backend already prepared a no-photo room with missing weights; weights alone did not reproduce all reported failures. The room worker still needed a narrow guard for unexpected vision exceptions.
- Pending launch navigation disabled Back, and restored pages needed loading/controller cleanup.

## Changes

Photo and validation state now belong to the wizard. Continue and launch have distinct DOM keys; launch is restricted to Step 4 and rejects repeated click events. Users can select a bundle, which is submitted as variation_id and applied by the room worker. Back aborts pending requests. Page restoration releases busy state. Vision initialization availability/inference exceptions use the existing catalogue and room-type fallback. Frontend artifacts were rebuilt and the wizard asset version changed.

## Evidence

- TypeScript typecheck and Vite production build passed.
- BackgroundRouteRegressions: 13 tests passed, including missing weights, inference exception, corrupt photo, selected bundle and 133-value state handoff.
- Live port 5000: double-click Step 3 Continue remained at Step 4; no room creation happened until launch.
- Uploaded data/samples/sample_room.jpg; Back then Continue retained its filename and image preview.
- Six bundles loaded. Selecting Modern Style match showed Selected for Studio.
- Final launch returned job HTTP 200 and opened Studio with INR 75,000 / 85,000, 100% A* reachability and zero collisions.
- Browser Back returned enabled controls. Back during a pending launch returned Step 3 without redirecting.
- Screenshot: output/step4-fixed.png.

Real YOLO inference is unverified because model weights are absent. Fallback behavior was tested. Browser history may reload the wizard; photo retention is guaranteed across the wizard's own step navigation, not full page reloads.

## Follow-up: photo inference enabled

Installed the vision runtime and auto-downloaded official yolov8n.pt. Corrected COCO name mapping and preserved photo results in recommendation/Studio responses. Live upload and browser evidence now confirm actual inference (two sample bed detections, 133-value state, no template fallback). Missing/corrupt photos still recover. Also fixed a separately observed history foreign-key failure caused by a stale user cookie. Current limits are pretrained class coverage and approximate photo-to-floor mapping, as documented in JOURNAL.md.
