# Verification journal

## 2026-09-29 — Viva refinement

User approved the proposed cleanup: obsolete room-details template, unused static sample copy, and 36 tracked bytecode files. No deletion script was written. Bytecode was removed from the index with local caches retained and ignored. Existing unrelated changes were preserved.

Evidence:

```text
.venv-viva/Scripts/python.exe -m unittest discover -s tests -q
Ran 30 tests in 5.662s
OK

AST documentation coverage (SAT, A*, DQN, environment, YOLO):
Class/function docstrings: 61 / 61

pnpm run typecheck
$ tsc --noEmit
exit code 0

pnpm run build
427 modules transformed
smartspace-frontend.css 19.69 kB
smartspace-frontend.js 339.88 kB
built in 1.68s

git diff --check: no whitespace errors
git ls-files '*__pycache__*': no tracked cache files
git check-ignore --no-index -v: confirmed requested ignore patterns,
plus NumPy checkpoints, generic .ckpt files and .venv-viva.

Live HTTP: POST /api/recommendations -> 202 in 30.5 ms
Concurrent GET /api/system-status -> 200
Job result: six bundles, all collision-free, five fully valid
Elapsed recommendation time: 0.28s (untrained fallback configuration)

Final default fallback (12x10 ft, Living Room, INR 85000):
collision_count=0, boundary_violations=0.0, door_interferences=0,
window_blockages=0, circulation_ratio=1.0, ergonomics_score=100.0
```

Browser validation used the built frontend at `127.0.0.1:5055`: all four wizard steps, loading status, six bundles, named-room transfer to Studio, optimization completing with zero collisions, and zero captured console warnings/errors. Screenshots and accessibility snapshots were inspected in the browser tool.

Limitations: no real YOLO/DQN checkpoints installed; real inference accuracy is unverified. Job state is process-local and requires a shared queue/store for multi-process deployment. Existing `.venv`/`venv` executables reference another computer, so validation used the ignored `.venv-viva` created here.

## 2026-09-30 — Step 4 repair

Fixed wizard photo retention, validation recovery, recommendation selection, distinct Continue/launch button identity, request cancellation and restored-page loading cleanup. The room worker now catches vision failures and honors the selected bundle during fallback. Rebuilt static/dist and bumped wizard cache version. See DEBUG.md for findings and evidence.

Validation: TypeScript and Vite build passed; 13 background route regression tests passed. Live browser verified Step 3 double-click, photo Back/Continue retention, six selectable bundles, selected INR 75,000 layout opening Studio, browser Back and cancellation. Flask remains running on 127.0.0.1:5000. Real YOLO inference still requires weights.

## 2026-09-30 — Real YOLO photo inference

Implemented automatic official yolov8n checkpoint loading, process-level loading lock, existing per-detector inference lock, COCO label-to-SmartSpace mapping, EXIF-corrected bounded JPG/PNG decoding, and multipart Step 4 submission to /api/detect. Both photo routes now retain photo-derived room state rather than discarding detections. Existing history/state preparation is reused. A stale login referencing a removed user is cleared before history saving, fixing the observed foreign-key 500. The annotated image uses contain sizing and the photo panel expands to avoid clipping. Added requirements-vision.txt and updated installation guidance. Existing offline test fixtures explicitly disable automatic downloads; no new test suite was run in this implementation turn.

Installed ultralytics 8.4.166, torch 2.14.0 and torchvision 0.29.0 into .venv-viva. Ultralytics downloaded the official 6.2 MB yolov8n.pt. Its settings directory is inside data/models/ultralytics and generated settings/weights are ignored. Restarted the old local server so the active site uses this environment.

Evidence:

```text
Python syntax OK
TypeScript: tsc --noEmit, exit 0
Vite: 427 modules; built in 2.58s; JS 340.95 kB
GET /api/system-status: available=true, backend=ultralytics, model=yolov8n.pt
POST sample_room.jpg /api/detect: 202 -> 200
detected_count=2; classes=[bed,bed]; fallback=false; state_length=133
Photo recommendations: HTTP 200, six bundles, two detections, state_length=133
No-photo and corrupt-photo requests: HTTP 200, fallback=true, state_length=133
Stale-cookie request: HTTP 200, success=true, state_length=133
```

Browser: uploaded the sample through Step 4, launched Studio, observed two detected bed entries and the annotated image labeled Real detections / ultralytics model. Inspected the final complete photo panel. Screenshot: output/yolo-photo-working.png. Flask remains on port 5000.

Limitations: stock COCO weights cover bed/sofa/chair/table/tv; other project classes need custom weights. Photo-to-floor coordinates and rotations are approximate, and the imported photo layout can need manual/DQN refinement. No claim of measured model accuracy or perfect spatial reconstruction.

## Compact furniture panel

Added frontend/src/components/FurniturePanel.tsx and mounted it in the existing Studio inventory slot. Cards use p-3, grouped status/name/dimensions, compact rotate/swap/delete controls, and grid-cols-1/md:grid-cols-2/xl:grid-cols-3. The wrapper hides overflow; the inner grid has max-h-64 and overflow-y-auto. Removed the legacy DOM card writer and nested scrolling wrapper rules. Connected React inventory snapshots/actions to the existing Canvas controller. Swap opens the existing catalog with an explicit target and cancel control, then replaces that item using the normal layout/metric updates. Updated template cache versions and rebuilt static/dist.

Evidence: TypeScript check exit 0; node --check static/js/app.js exit 0; Vite transformed 428 modules and built in 1.92s. Live computed styles: outerOverflow=hidden, gridOverflow=auto, maxHeight=256px, three grid columns of 364px. Browser showed three compact cards; Swap opened the correct replacement prompt, then Cancel returned to the unchanged inventory. Screenshot: output/furniture-panel-compact.png. No new automated tests were added or run.
